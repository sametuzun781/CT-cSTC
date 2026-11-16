"""Portable CPU discretization for the supplied rocket model.

Runtime parameters are passed to the generated field; they are not baked into
its numerical coefficients. Native build, input/output transfers and kernel
execution are distinct. Public __call__ includes all transfers and allocation.
"""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
import jaxlib
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'native_discretization'
_LIBRARIES={}


def eligibility(params):
    if jax.default_backend() != 'cpu' or not jax.config.x64_enabled:
        return False, 'Native discretization requires CPU and float64.'
    if not (params['n_x'] == 15 and params['n_u'] == 5 and params['free_final_time']
            and params['time_dil'] and params['inp_param'] == 'FOH'):
        return False, 'Native discretization requires the supplied FOH/time-dilation model.'
    callbacks = tuple(params[name] for name in ('f_func', 'A_func', 'B_func'))
    certified = params.get('_native_rocket_functions', ())
    if len(certified) != 3 or any(a is not b for a, b in zip(callbacks, certified)):
        return False, 'Dynamics callbacks were replaced; use JAX or regenerate the native field.'
    book = json.loads((ROOT / 'CT-cSTC.ipynb').read_text())
    sources = [''.join(cell['source']) for cell in book['cells'] if cell['cell_type'] == 'code']
    field_source = '\n'.join(next(s for s in sources if s.startswith(prefix))
                             for prefix in ('def euler_to_quat(', 'def rl_dynamics('))
    layout = json.loads((SOURCE / 'field_layout.json').read_text())
    if hashlib.sha256(field_source.encode()).hexdigest() != layout['source_sha256']:
        return False, 'Rocket source changed; use JAX or regenerate the native field.'
    return True, ''


def make_discretizer(params, reference_factory, mode='fast', compile=True):
    if mode not in ('fast', 'native', 'jax', 'reference'):
        raise ValueError('DISCRETIZATION_MODE must be fast, native, jax, or reference.')
    if mode in ('fast', 'native'):
        valid, reason = eligibility(params)
        if valid:
            try:
                return NativeRocketDiscretizer(params)
            except (OSError, RuntimeError) as error:
                if mode == 'native':
                    raise
                import warnings
                warnings.warn(f'Native build unavailable; using JAX: {error}', RuntimeWarning)
        elif mode == 'native':
            raise ValueError(reason)
    function = reference_factory(params, compile=compile)
    function.discretization_backend = 'reference' if mode == 'reference' else 'jax'
    return function


@lru_cache(maxsize=8)
def build_library():
    compiler=shlex.split(os.environ.get('CC','cc'))
    cxx=shlex.split(os.environ.get('CXX','c++'))
    for command in (compiler,cxx):
        if not command or not shutil.which(command[0]):
            raise RuntimeError('Native rocket discretization needs C/C++ compilers (macOS: xcode-select --install).')
    arch=['-march=native'] if platform.machine().lower() in ('x86_64','amd64') else ['-mcpu=native'] if platform.machine().lower() in ('arm64','aarch64') else []
    if os.environ.get('ICORE_DISCRETIZATION_ARCH','native')=='portable': arch=[]
    versions=[subprocess.check_output([*c,'--version'],text=True).splitlines()[0] for c in (compiler,cxx)]
    key=hashlib.sha256(b''.join((SOURCE/name).read_bytes() for name in ['rocket_rk4.c','rocket_field.c','sensitivity_plan.h','rocket_ffi.cc'])
                       +repr((compiler,cxx,versions,sys.platform,platform.machine(),arch,jaxlib.__version__)).encode()).hexdigest()[:20]
    cache=ROOT/'.build/rocket_discretization';cache.mkdir(parents=True,exist_ok=True)
    output=cache/('librocket_'+key+('.dylib' if sys.platform=='darwin' else '.so'))
    if output.exists():return output
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        temp=Path(directory)/output.name;obj=Path(directory)/'rocket.o'
        commands=[[*compiler,'-O3',*arch,'-std=c11','-fPIC','-c',str(SOURCE/'rocket_rk4.c'),'-o',str(obj)],
                  [*cxx,'-O3','-std=c++17','-fPIC','-dynamiclib' if sys.platform=='darwin' else '-shared',
                   '-I'+jax.ffi.include_dir(),str(SOURCE/'rocket_ffi.cc'),str(obj),'-o',str(temp),'-lm']]
        log=[]
        for command in commands:
            result=subprocess.run(command,text=True,capture_output=True)
            log.append(' '.join(command)+'\n'+result.stdout+result.stderr)
            (cache/'build.log').write_text('\n'.join(log))
            if result.returncode:raise RuntimeError('Rocket discretizer build failed:\n'+result.stderr)
        os.replace(temp,output)
    return output


class NativeRocketDiscretizer:
    """Exact sparse field/Jacobian evaluation followed by the original RK4 grid."""
    discretization_backend = 'native'
    def __init__(self,params,eager=True):
        if not (params['n_x']==15 and params['n_u']==5 and params['free_final_time']
                and params['time_dil'] and params['inp_param']=='FOH'):
            raise ValueError('Native rocket discretizer requires the supplied FOH/time-dilation model.')
        self.nodes=int(params['K']);self.points=int(params['rk4_steps_J'])
        if self.nodes<2 or self.points<2:raise ValueError('At least two nodes and two RK4 points are required.')
        self.layout=json.loads((SOURCE/'field_layout.json').read_text())
        self.parameters=np.empty(self.layout['parameter_count'],dtype=np.float64)
        for row in self.layout['parameters']:
            a=np.asarray(params[row['key']],dtype=np.float64)
            if list(a.shape)!=row['shape'] or not np.isfinite(a).all():
                raise ValueError('Invalid physical parameter: '+row['key'])
            self.parameters[row['offset']:row['offset']+row['size']]=a.ravel()
        self.library=None;self.compile_seconds=0.0;self.last_timing={}
        if eager:self.prepare()

    def prepare(self):
        if self.library is not None:return
        if jax.default_backend()!='cpu' or not jax.config.x64_enabled:
            raise ValueError('Native discretization requires the CPU backend and JAX float64; use reference mode otherwise.')
        start=perf_counter();path=build_library();library=ct.CDLL(str(path))
        vector=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        library.icore_rocket_discretize.argtypes=[ct.c_int,ct.c_int,vector,vector,vector,vector,vector]
        library.icore_rocket_discretize.restype=ct.c_int
        library.rocket_field.argtypes=[vector,vector,vector,vector,vector]
        library.rocket_field.restype=None
        target='icore_rocket_rk4_'+path.stem
        if target not in _LIBRARIES:
            jax.ffi.register_ffi_target(target,jax.ffi.pycapsule(library.ICORERocketRk4),platform='cpu')
            _LIBRARIES[target]=library
        config=jnp.asarray([self.nodes,self.points,1,1,0,self.parameters.size],dtype=jnp.int32)
        physical=jnp.asarray(self.parameters)
        call=jax.ffi.ffi_call(target,jax.ShapeDtypeStruct((self.nodes-1,420),np.float64),vmap_method='sequential')
        def discretize(X,U,S):
            out=call(X.T,U.T,S.reshape(-1),physical,config,jnp.asarray(1.,jnp.float64))
            return dict(f_bar=out[:,:15].T,
              A_bar=out[:,15:240].reshape(-1,15,15).transpose(0,2,1).reshape(-1,225).T,
              B_bar=out[:,240:315].reshape(-1,15,5).transpose(0,2,1).reshape(-1,75).T,
              C_bar=out[:,315:390].reshape(-1,15,5).transpose(0,2,1).reshape(-1,75).T,
              S_bar=out[:,390:405].T,z_bar=out[:,405:420].T)
        self.jax_function=jax.jit(discretize)
        self._compiled=self.jax_function.lower(jax.ShapeDtypeStruct((15,self.nodes),np.float64),
            jax.ShapeDtypeStruct((5,self.nodes),np.float64),jax.ShapeDtypeStruct((1,self.nodes-1),np.float64)).compile()
        self.library=library
        self.compile_seconds+=perf_counter()-start

    def field_and_jacobian(self,x,u):
        self.prepare();x=np.ascontiguousarray(x,dtype=np.float64);u=np.ascontiguousarray(u,dtype=np.float64)
        if x.shape!=(15,) or u.shape!=(5,):raise ValueError('Expected x[15], u[5].')
        value=np.empty(15);entries=np.empty(len(self.layout['jacobian_indices']))
        self.library.rocket_field(x,u,self.parameters,value,entries)
        jac=np.zeros((15,20))
        for (i,j),v in zip(self.layout['jacobian_indices'],entries):jac[i,j]=v
        return value,jac

    def __call__(self,X,U,S):
        self.prepare()
        started=perf_counter()
        X,U,S=(value if isinstance(value,jax.Array) and value.dtype==np.float64
               else jnp.asarray(value,dtype=jnp.float64) for value in (X,U,S))
        if S.shape!=(1,self.nodes-1):S=S.reshape(1,-1)
        if X.shape!=(15,self.nodes) or U.shape!=(5,self.nodes) or S.shape!=(1,self.nodes-1):
            raise ValueError('Trajectory shapes do not match the prepared discretizer.')
        result=self._compiled(X,U,S)
        jax.block_until_ready(result)
        self.last_timing=dict(total_seconds=perf_counter()-started,
            timing_scope='Complete synchronized JAX/FFI call, including layout changes and output formation')
        return result

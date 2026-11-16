"""Generate exact symbolic sparse derivatives from the notebook's rocket field.

Optional development tool; SymPy is NOT a runtime dependency of the examples.
"""
from pathlib import Path
import ast, json, sys, typing as T, hashlib
import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ctcstc.application import load_application
_,numeric,_=load_application(discretization="reference")
nb=json.loads((ROOT/'CT-cSTC.ipynb').read_text())
sources=[''.join(cell['source']) for cell in nb['cells'] if cell['cell_type']=='code']
utility=next(s for s in sources if s.startswith('def euler_to_quat('))
source=next(s for s in sources if s.startswith('def rl_dynamics('))
class Indexed:
 def __init__(self,array,index):self.array,self.index=array,index
 def set(self,value):
  arr=self.array.copy();arr[self.index]=value;return arr
 def add(self,value):
  arr=self.array.copy();arr[self.index]+=value;return arr
class At:
 def __init__(self,array):self.array=array
 def __getitem__(self,index):return Indexed(self.array,index)
class SymArray(np.ndarray):
 @property
 def at(self):return At(self)
class SymbolicMath:
 def array(self,v,*args,**kwargs):return np.array(v,dtype=object).view(SymArray)
 def zeros(self,shape,*args,**kwargs):return np.zeros(shape,dtype=object).view(SymArray)
 def dot(self,a,b):return np.dot(a,b)
 def sin(self,a):return sp.sin(a)
 def cos(self,a):return sp.cos(a)
 def tan(self,a):return sp.tan(a)
 def sqrt(self,a):return sp.sqrt(a)
 def deg2rad(self,a):return a*(sp.pi/180)
 def maximum(self,a,b):return sp.Max(a,b)
 pi=sp.pi
 def __getattr__(self,key):return getattr(np,key)
sym=SymbolicMath()
keys=sorted({n.slice.value for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='params' and isinstance(n.slice,ast.Constant)}-{'n_x','f_func','A_func','B_func','_native_rocket_functions'})
params=numeric.copy();layout=[];offset=0
for key in keys:
 val=np.asarray(numeric[key]);shape=val.shape;size=val.size
 symbolic=sym.array([sp.Symbol(f'p[{offset+i}]',real=True) for i in range(size)]).reshape(shape)
 params[key]=symbolic.item() if not shape else symbolic
 layout.append(dict(key=key,shape=list(shape),offset=offset,size=int(size)));offset+=size
ns=dict(np=sym,jnp=sym,T=T,jit=lambda f:f,jacfwd=lambda *a,**k:None,jacrev=lambda *a,**k:None,DYNAMICS_AD='forward')
exec(utility,ns);exec(source,ns)
f=ns['rl_dynamics'](params)['f_func']
x=sym.array([sp.Symbol(f'x[{i}]',real=True) for i in range(15)])
u=sym.array([sp.Symbol(f'u[{i}]',real=True) for i in range(5)])
values=list(f(x,u));print('SYMBOLIC FIELD READY',flush=True)
jac=[];indices=[]
for i,expr in enumerate(values):
 for j,var in enumerate([*x,*u]):
  d=sp.diff(expr,var)
  if d!=0:jac.append(d);indices.append((i,j))
 print('DERIVATIVE ROW',i,'NNZ',len(jac),flush=True)
replacements,reduced=sp.cse(values+jac,symbols=sp.numbered_symbols('t'),order='canonical')
print('CSE',len(replacements),len(reduced),flush=True)
class Printer(C99CodePrinter):
 def _print_Heaviside(self,expr):
  value=self._print(expr.args[0]);return f'(({value}) > 0.0 ? 1.0 : (({value}) < 0.0 ? 0.0 : 0.5))'
 def _print_Pow(self,expr):
  a,b=expr.args
  if b==sp.Float(.5):return f'sqrt({self._print(a)})'
  if b==sp.Float(-.5):return f'(1.0 / sqrt({self._print(a)}))'
  return super()._print_Pow(expr)
printer=Printer()
lines=['/* Generated from the notebook field. Do not hand-edit derivative entries. */','#include <math.h>','#include <stddef.h>',f'#define ROCKET_NNZ {len(indices)}',f'#define ROCKET_NPARAM {offset}',
 'static const int rocket_jrow[ROCKET_NNZ] = {'+','.join(str(i) for i,j in indices)+'};',
 'static const int rocket_jcol[ROCKET_NNZ] = {'+','.join(str(j) for i,j in indices)+'};',
 'void rocket_field(const double *x, const double *u, const double *p, double *f, double *j) {']
for lhs,rhs in replacements:lines.append(f'    const double {lhs} = {printer.doprint(rhs)};')
for i,expr in enumerate(reduced):lines.append(f'    {"f" if i<15 else "j"}[{i if i<15 else i-15}] = {printer.doprint(expr)};')
lines.append('}')
out=ROOT/'native_discretization';out.mkdir(exist_ok=True)
(out/'rocket_field.c').write_text('\n'.join(lines)+'\n')
(out/'field_layout.json').write_text(json.dumps(dict(parameters=layout,parameter_count=offset,jacobian_indices=indices,
 source_sha256=hashlib.sha256((utility+'\n'+source).encode()).hexdigest(),f_dimension=15,u_dimension=5),indent=2))
print('GENERATED',out,flush=True)

from finish_native_generation import finish
finish(ROOT,"rocket",15,5)

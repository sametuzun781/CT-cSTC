"""Final, reproducible postprocessing of generated analytic dynamics."""
from pathlib import Path
import re
from hoist_parameters import hoist
from generate_transport import generate

def finish(root, model, nx, nu):
    path=Path(root)/'native_discretization'/f'{model}_field.c'
    hoist(path,model)
    text=path.read_text()
    helper="static inline double positive_part(double x) { return isnan(x)?x:(x>0.0?x:0.0); }"
    if model=='quad':
        helper+="\nstatic inline double negative_part(double x) { return isnan(x)?x:(x<0.0?x:0.0); }"
    else:
        helper="/* Positive part preserves NaN propagation of the source JAX expression. */\n"+helper
    text=text.replace('#include <stddef.h>','#include <stddef.h>\n'+helper)
    text=re.sub(r'fmax\(0(?:\.0)?, (t\d+)\)',r'positive_part(\1)',text)
    if model=='quad':text=re.sub(r'fmin\(0(?:\.0)?, (t\d+)\)',r'negative_part(\1)',text)
    path.write_text(text)
    generate(root,model,nx,nu)

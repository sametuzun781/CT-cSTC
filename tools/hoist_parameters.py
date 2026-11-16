"""Hoist state/control-independent CSE nodes out of the RK4 loop, exactly."""
import re,sys
from pathlib import Path

def hoist(path,model):
 text=Path(path).read_text(); prefix,body=text.split(f'void {model}_field(',1);body=body[body.index('{')+1:body.rindex('}')]
 invariant={}; prepare=[]; dynamic=[]
 for line in body.splitlines():
  m=re.fullmatch(r'\s*const double (t\d+) = (.*);',line)
  if m:
   name,expr=m.groups();refs=re.findall(r'\bt\d+\b',expr)
   expr=re.sub(r'\bt\d+\b',lambda v:invariant.get(v[0],v[0]),expr)
   if not re.search(r'\b[ux]\[',expr) and all(t in invariant for t in refs):
    q=f'q[{len(invariant)}]';invariant[name]=q;prepare.append(f'    {q} = {expr};')
   else:dynamic.append(f'    const double {name} = {expr};')
  elif line.strip():dynamic.append(re.sub(r'\bt\d+\b',lambda v:invariant.get(v[0],v[0]),line))
 n=len(invariant);upper=model.upper()
 out=prefix+f'#define {upper}_NINVARIANT {n}\n'+f'static void {model}_prepare(const double *p, double *q) {{\n'+'\n'.join(prepare)+'\n}\n'
 out+=f'static void {model}_field_prepared(const double *x, const double *u, const double *p, double *f, double *j) {{\n'+f'    const double *q=p+{upper}_NPARAM;\n'+'\n'.join(dynamic)+'\n}\n'
 out+=f'void {model}_field(const double *x, const double *u, const double *p, double *f, double *j) {{\n'+f'    double prepared[{upper}_NPARAM+{upper}_NINVARIANT];\n'+f'    for(int i=0;i<{upper}_NPARAM;++i) prepared[i]=p[i];\n'+f'    {model}_prepare(p,prepared+{upper}_NPARAM);\n'+f'    {model}_field_prepared(x,u,prepared,f,j);\n}}\n'
 Path(path).write_text(out);print(model,'hoisted',n)
if __name__=='__main__':hoist(sys.argv[1],sys.argv[2])

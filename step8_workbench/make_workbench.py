# inlines workbench_data.js into workbench_template.html -> workbench.html (one file, no server)
import os
H=os.path.dirname(os.path.abspath(__file__))
t=open(os.path.join(H,'workbench_template.html')).read()
d=open(os.path.join(H,'workbench_data.js')).read()
lj=open(os.path.join(H,'vendor','leaflet.js')).read(); lc=open(os.path.join(H,'vendor','leaflet.css')).read()
t=t.replace('/*LEAFLET_CSS*/',lc).replace('/*LEAFLET_JS*/',lj)
open(os.path.join(H,'workbench.html'),'w').write(t.replace('/*DATA*/',d))
print('wrote workbench.html', os.path.getsize(os.path.join(H,'workbench.html')),'bytes')

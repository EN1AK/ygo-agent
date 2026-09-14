import sys, runpy
from pathlib import Path
root=Path('/root/ygo-agent-gpu-20260910/ygo-agent')
area=root/'reports/multideck-20260912'
sys.path.insert(0,str(root))
import ygoai
ygoai.__path__.append(str(area/'overlay/ygoai'))
sys.path.insert(0,str(root/'scripts'))
runpy.run_path(str(area/'overlay/scripts/cleanba.py'),run_name='__main__')

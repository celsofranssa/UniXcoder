# activate venv and set Python path
source ~/projects/venvs/UniXcoder/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/UniXcoder/

python main.py \
  tasks=[fit,predict,eval] \
  model=UNIX \
  data=PYTHON

python main.py \
  tasks=[fit,predict,eval] \
  model=UNIX \
  data=JAVASCRIPT


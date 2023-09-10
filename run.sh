# activate venv and set Python path
source ~/projects/venvs/UniXcoder/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/UniXcoder/

python main.py \
  tasks=[fit,predict,eval] \
  model=CodeBERT \
  data=PYTHON \
  data.folds=[0,1,2,3,4]

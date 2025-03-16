gpuid=1
N_SHOT=5
DATA_ROOT=filelist/SSv2Full
MODEL_PATH=/hd1/wyl/model/112112vit-s-140epoch.pt
YOURPATH=xxx/xxx/xxx   # PATH of your CKPT
cd ../../../

# echo "============= meta-train 5-shot ============="
# python meta_train.py --dataset SSv2Full --data_path $DATA_ROOT  --model VideoMAES2 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 40 --milestones 40 --n_shot $N_SHOT --train_n_episode 600 --val_n_episode 300 --reduce_dim 256 --pretrain_path $MODEL_PATH >> $YOURPATH/trainlog.txt

echo "============= meta-train 5-shot ============="
python meta_train.py --dataset SSv2Full --data_path $DATA_ROOT  --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --lr 5e-4 --epoch 30 --milestones 30 --n_shot $N_SHOT --train_n_episode 600 --val_n_episode 300 --reduce_dim 256 --pretrain_path $MODEL_PATH >> $YOURPATH/trainlog.txt

echo "============= meta-test best_model ============="
MODEL_PATH=$YOURPATH/best_model.tar
python test.py --dataset SSv2Full --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --n_query 12 --test_n_episode 1000 --model_path $MODEL_PATH --test_task_nums 5 >> $YOURPATH/FFTucf_ssv2bestlog.txt

echo "============= meta-test last_model ============="
MODEL_PATH=$YOURPATH/last_model.tar
python test.py --dataset SSv2Full --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --n_query 12 --test_n_episode 1000 --model_path $MODEL_PATH --test_task_nums 3 >> $YOURPATH/lastlog.txt
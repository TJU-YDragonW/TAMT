gpuid=3
N_SHOT=5
DATA_ROOT=filelist/ucf101-molo
MODEL_PATH=/hd1/wyl/model/112112vit-s-140epoch.pt
YOURPATH=xxx/xxx/xxx   # PATH of your CKPT
cd ../../../

echo "============= meta-train 5-shot ============="
python meta_train.py --dataset ucf101 --data_path $DATA_ROOT  --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --lr 1e-3 --epoch 25 --milestones 25 --n_shot $N_SHOT --train_n_episode 300 --val_n_episode 150 --reduce_dim 256 --pretrain_path $MODEL_PATH >> $YOURPATH/trainlog.txt

echo "============= meta-test best_model ============="
MODEL_PATH=$YOURPATH/best_model.tar
python test.py --dataset ucf101 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 1 >> $YOURPATH/bestlog.txt

echo "============= meta-test last_model ============="
MODEL_PATH=$YOURPATH/last_model.tar
python test.py --dataset ucf101 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 >> $YOURPATH/lastlog.txt
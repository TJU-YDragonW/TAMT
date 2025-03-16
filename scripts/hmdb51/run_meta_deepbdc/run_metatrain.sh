gpuid=0
N_SHOT=5
DATA_ROOT=filelist/hmdb51-molo
MODEL_PATH=/hd1/wyl/model/112112vit-s-140epoch.pt     # PATH of your Pretrained MODEL
YOURPATH=xxx/xxx/xxx  # PATH of your CKPT, e.g., Mine: /home/wyll/TAMT/checkpoints/hmdb51/VideoMAES_meta_deepbdc_5way_5shot_2TAA
cd ../../../


echo "============= meta-train 5-shot ============="

# # train with log, 112 resolution
# python meta_train.py --dataset hmdb51 --data_path $DATA_ROOT  --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --lr 1e-3  --epoch 30 --milestones 30 --n_shot $N_SHOT --train_n_episode 600 --val_n_episode 300  --reduce_dim 256 --pretrain_path $MODEL_PATH >> $YOURPATH/trainlog.txt

# train without log, 112 resolution
python meta_train.py --dataset hmdb51 --data_path $DATA_ROOT  --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --lr 1e-3  --epoch 30 --milestones 30 --n_shot $N_SHOT --train_n_episode 600 --val_n_episode 300  --reduce_dim 256 --pretrain_path $MODEL_PATH #>> $YOURPATH/trainlog.txt

# # 224 resolution
# python meta_train.py --dataset hmdb51 --data_path $DATA_ROOT  --model VideoMAES2 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --lr 5e-4  --epoch 60 --milestones 20 --n_shot $N_SHOT --train_n_episode 600 --val_n_episode 300  --reduce_dim 256 --pretrain_path $MODEL_PATH #>> $YOURPATH/trainlog.txt

echo "============= meta-test best_model ============="
MODEL_PATH=$YOURPATH/best_model.tar
python test.py --dataset hmdb51 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT  --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 #>> $YOURPATH/bestlog.txt

echo "============= meta-test last_model ============="
MODEL_PATH=$YOURPATH/last_model.tar
python test.py --dataset hmdb51 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 #>> $YOURPATH/lastlog.txt
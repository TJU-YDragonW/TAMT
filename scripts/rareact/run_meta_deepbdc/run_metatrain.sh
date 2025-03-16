gpuid=2
N_SHOT=5
DATA_ROOT=filelist/rareact_cut
MODEL_PATH=/hd1/wyl/model/112112vit-s-140epoch.pt
YOURPATH=xxx/xxx/xxx  # PATH of your CKPT

cd ../../../

# echo "============= meta-train 5-shot ============="
# python meta_train.py --dataset Rareact2 --data_path $DATA_ROOT  --model VideoMAES2 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --lr 1e-5 --epoch 30 --milestones 20 --n_shot $N_SHOT --n_query 1 --train_n_episode 600 --val_n_episode 500 --reduce_dim 256 --pretrain_path $MODEL_PATH >> $YOURPATH/trainlog.txt

echo "============= meta-train 5-shot ============="
python meta_train.py --dataset Rareact2 --data_path $DATA_ROOT  --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --lr 1e-6 --epoch 40 --milestones 20 --n_shot $N_SHOT --n_query 1 --train_n_episode 600 --val_n_episode 500 --pretrain_path $MODEL_PATH >> $YOURPATH/trainlog.txt

echo "============= meta-test best_model ============="
MODEL_PATH=$YOURPATH/best_model.tar
python test500.py --dataset Rareact2 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --n_query 1 --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 >> $YOURPATH/bestlog.txt

echo "============= meta-test best_10model ============="
MODEL_PATH=$YOURPATH/10.tar
python test500.py --dataset Rareact2 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --n_query 1 --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 3 >> $YOURPATH/10.txt

echo "============= meta-test 20 ============="
MODEL_PATH=$YOURPATH/20.tar
python test500.py --dataset Rareact2 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --n_query 1 --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 3 >> $YOURPATH/20.txt
echo "============= meta-test last_model ============="
MODEL_PATH=$YOURPATH/last_model.tar
python test500.py --dataset Rareact2 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --n_query 1 --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 >> $YOURPATH/lastlog.txt


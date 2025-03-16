gpuid=9
N_SHOT=5
DATA_ROOT=filelist/diving48
MODEL_PATH=/hd1/wyl/model/112112vit-s-140epoch.pt

cd ../../../

echo "============= meta-train 5-shot ============="
python meta_train.py --dataset diving48 --data_path $DATA_ROOT  --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --lr 1e-3 --epoch 25 --milestones 25 --n_shot $N_SHOT --train_n_episode 300 --val_n_episode 150 --reduce_dim 256 --pretrain_path $MODEL_PATH #>> /home/wyll/TAMT/checkpoints/diving48/VideoMAES_meta_deepbdc_5way_5shot_2TAA/trainlog.txt

# echo "============= meta-train 5-shot ============="
# python meta_train.py --dataset diving48 --data_path $DATA_ROOT  --model VideoMAES2 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --lr 5e-4 --epoch 50 --milestones 40 --n_shot $N_SHOT --train_n_episode 30 --val_n_episode 15 --reduce_dim 256 --pretrain_path $MODEL_PATH >> YOURPATH/trainlog.txt

echo "============= meta-test best_model ============="
MODEL_PATH=/home/wyll/TAMT/checkpoints/diving48/VideoMAES_meta_deepbdc_5way_5shot_2TAA/best_model.tar
python test.py  --dataset diving48 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 3 >> /home/wyll/TAMT/checkpoints/diving48/VideoMAES_meta_deepbdc_5way_5shot_2TAA/bestlog.txt

echo "============= meta-test last_model ============="
MODEL_PATH=/home/wyll/TAMT/checkpoints/diving48/VideoMAES_meta_deepbdc_5way_5shot_2TAA/last_model.tar
python test.py  --dataset diving48 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 3 >> /home/wyll/TAMT/checkpoints/diving48/VideoMAES_meta_deepbdc_5way_5shot_2TAA/lastlog.txt
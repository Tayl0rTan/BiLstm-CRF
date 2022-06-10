export BERT_BASE_DIR=/home/hadoop-mtai/cephfs/data/tanxiaoyu02/bert
export GLUE_DIR=/data/hadoop-mtai/zhouhuajie04/bert_example_data

python main.py \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=./data \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=16 \
--learning_rate=1e-5 \
--num_train_epochs=100 \
--output_dir=./model/

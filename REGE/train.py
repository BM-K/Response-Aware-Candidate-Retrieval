import time
from trainer.setting import Setting, Arguments
from trainer.GenerationModel.processor import Processor
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args, logger) -> None:

    processor = Processor(args)
    config = processor.model_setting()
    logger.info('Model Setting Complete')

    if args.train == 'True':
        logger.info('Start Training')

        for epoch in range(args.epochs):
            start_time = time.time()

            train_loss = processor.train()
            valid_loss = processor.valid()

            end_time = time.time()
            epoch_mins, epoch_secs = processor.metric.cal_time(start_time, end_time)

            performance = {'tl': train_loss, 'vl': valid_loss,
                           'ep': epoch, 'epm': epoch_mins, 'eps': epoch_secs}

            processor.metric.save_model(config, performance, processor.model_checker)

            if processor.model_checker['early_stop']:
                logger.info('Early Stopping')
                break

    if args.test == 'True':
        logger.info("Start Test")

        rouge_score, bleu_score, dist_score = processor.test()
        print(f'\n{bleu_score}')
        print(f'\n{dist_score}')
        print(f'\n{rouge_score}')

        path_to_save_test_score = args.path_to_save + args.ckpt.split('.')[0] + '_test_score.txt'
        with open(path_to_save_test_score, "w", encoding="utf-8") as file:
            file.write(str(bleu_score))
            file.write(str(dist_score))
            file.write(str(rouge_score))

        processor.metric.print_size_of_model(config['model'])


if __name__ == '__main__':
    args, logger = Setting().run()
    main(args, logger)

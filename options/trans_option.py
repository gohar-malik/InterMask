from options.base_option import BaseOptions
import argparse

class TrainTransOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=52, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epoch for training')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--step_unroll', type=float, default=1, help='Step Unroll masking factor')
        self.parser.add_argument('--interaction_mask_prob', type=float, default=0.2, help='Interaction Mask probability')
        self.parser.add_argument('--gamma', type=float, default=1/3, help='Learning rate schedule factor')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')

        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
        
        '''eval'''
        self.parser.add_argument('--do_eval', action="store_true", help='Perform evaluations during training')
        self.parser.add_argument('--test_batch_size', default=96, type=int, help='batch size for evaluation')


        self.is_train = True

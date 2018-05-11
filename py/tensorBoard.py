import sys

import tensorboard.main

import conf

sys.argv += ["--logdir", conf.log_dir]
tensorboard.main.run_main()

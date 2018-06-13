import sys

import tensorboard.main

import conf

sys.argv += ["--logdir", conf.get_log_dir(conf.default_version)]
tensorboard.main.run_main()

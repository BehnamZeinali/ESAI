#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 03:10:55 2024

@author: behnam
"""

import subprocess
script_path_0 = '.python/server_model.py'
script_path_1 = '.python/get_logits.py'
script_path_2 = '.python/distillation_nas.py'
script_path_3 = '.python/decision_unit.py'
subprocess.run(['python', script_path_0])
subprocess.run(['python', script_path_1])
subprocess.run(['python', script_path_2])
subprocess.run(['python', script_path_3])
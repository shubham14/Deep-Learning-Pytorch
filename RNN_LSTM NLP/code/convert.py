# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:20:16 2019

@author: Shubham
"""

def convert(file_name, target_file):
    f1 = open(target_file, 'w')
    with open(file_name, 'r') as f:
          lines = f.readlines()
          lines = [line.strip() for line in lines]
    s_lines = list(map(lambda x: str(x[0]), lines))
    for s_line in s_lines:
        f1.write(s_line)
        f1.write('\n')
        
if __name__ == "__main__":
    convert('predictions_q1_1.txt', 'prediction_q1.txt')
    convert('predictions_q2_1.txt', 'prediction_q2.txt')
    convert('predictions_q3_1.txt', 'prediction_q3.txt')
    convert('predictions_q4_1.txt', 'prediction_q4.txt')
    convert('predictions_q5_1.txt', 'prediction_q5.txt')
    
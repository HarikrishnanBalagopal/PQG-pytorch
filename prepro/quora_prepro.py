import os
import csv
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../data/quora_duplicate_questions.tsv', help='path to tab separated file (.tsv) containing paraphrases.')
    parser.add_argument('--dataset_name', type=str, default='quora', help='name of the dataset.')
    return parser.parse_args()

def main(args):
    out = []
    outtest = []
    outval = []
    with open(args.input_path,'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')#read the tsv file of quora question pairs
        count0 = 1
        count1 = 1
        counter = 1
        for row in tsvin:
            counter = counter+1
            if row[5]=='0': # and row[4][-1:]=='?':#the 6th entry in every row has value 0 or 1 and it represents paraphrases if that value is 1
                count0=count0+1
            elif row[5]=='1': # and row[4][-1:]=='?':
                count1=count1+1
                if count1>1 and count1<100002:#taking the starting 1 lakh pairs as train set. Change this to 50002 for taking staring 50 k examples as train set
                    # get the question and unique id from the tsv file
                    img_id = row[0] #unique id for every pair
                    quesid = row[1] #first question id
                    quesid1 =row[2]#paraphrase question id                
                    ques = row[3] #first question
                    ques1 = row[4]#paraphrase question
                    
                    # set the parameters of json file for writing 
                    jimg = {}

                    jimg['question'] = ques
                    jimg['question1'] = ques1
                    jimg['ques_id'] =  quesid
                    jimg['ques_id1'] =  quesid1
                    jimg['id'] =  img_id

                    out.append(jimg)

                elif count1>100001 and count1<130002:#next 30k as the test set acc to https://arxiv.org/pdf/1711.00279.pdf
                    img_id = row[0] 
                    quesid = row[1] 
                    quesid1 =row[2]
                    ques = row[3] 
                    ques1 = row[4]

                    jimg = {}

                    jimg['question'] = ques
                    jimg['question1'] = ques1
                    jimg['ques_id'] =  quesid
                    jimg['ques_id1'] =  quesid1
                    jimg['id'] =  img_id

                    outtest.append(jimg)    
                else :#rest as val
                    img_id = row[0] 
                    quesid = row[1] 
                    quesid1 =row[2]
                    ques = row[3] 
                    ques1 = row[4]
                
                    jimg = {}
                    jimg['question'] = ques
                    jimg['question1'] = ques1
                    jimg['ques_id'] =  quesid
                    jimg['ques_id1'] =  quesid1
                    jimg['id'] =  img_id
                    
                    outval.append(jimg)
    #write the json files for train test and val
    print(len(out))
    json.dump(out, open(f'../data/{args.dataset_name}_raw_train.json', 'w'))
    print(len(outtest))
    json.dump(outtest, open(f'../data/{args.dataset_name}_raw_test.json', 'w'))
    print(len(outval))
    json.dump(outval, open(f'../data/{args.dataset_name}_raw_val.json', 'w'))


if __name__ == "__main__":
    main(parse_args())

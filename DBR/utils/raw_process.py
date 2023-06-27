import json
import os
import sys
import jsonlines
import csv


class raw_processer(object):
    """
        raw data process
    """
    def __init__(self, dataset_name) -> None:
        self.datasetname = dataset_name
        self.dirname = os.path.join("../datasets", self.datasetname)

    def reader(self):
        """
            this function return orign file path
        """
        
        if self.datasetname == "mnli":
            
            train_file_path =  os.path.join(self.dirname , "train.jsonl")
            val_match_file_path = os.path.join(self.dirname , "validation_matched.jsonl")
            val_mis_file_path = os.path.join(self.dirname , "validation_mismatched.jsonl")
            test_match_file_path = os.path.join(self.dirname , "test_matched.jsonl")
            test_mis_file_path = os.path.join(self.dirname , "test_mismatched.jsonl")

            return {
                "tarin" : train_file_path,
                "val_match" : val_match_file_path,
                "val_mismatch" : val_mis_file_path,
                "test_match" : test_match_file_path,
                "test_mismatch" : test_mis_file_path
            }
        if self.datasetname == "hans":
            train_file_path = os.path.join(self.dirname , "heuristics_train_set.txt")
            test_file_path = os.path.join(self.dirname, "heuristics_evaluation_set.txt")

            return {
                "train":train_file_path,
                "test":test_file_path
            }
        if self.datasetname == "mnli_hard":
            match_dir = os.path.join(self.dirname , "multinli-matched-open-hard-evaluation")
            mismatch_dir = os.path.join(self.dirname , "multinli-mismatched-open-hard-evaluation")

            return {
                "match_dir":match_dir,
                "mismatch_dir":mismatch_dir
            }
        if self.datasetname == "fever":
            train_file_path = os.path.join(self.dirname, "fever.train.jsonl")
            dev_file_path = os.path.join(self.dirname, "fever.dev.jsonl")
            v1_file_path = os.path.join(self.dirname, "fever_symmetric_v1.jsonl")
            v2_file_path = os.path.join(self.dirname, "fever_symmetric_v2.jsonl")
            return {
                "train": train_file_path,
                "dev": dev_file_path,
                "v1": v1_file_path,
                "v2": v2_file_path
            }
        if self.datasetname == "QQP":
            train_file_path = os.path.join(self.dirname, "train.tsv")
            dev_file_path = os.path.join(self.dirname, "dev.tsv")
            paws_file_path = os.path.join(self.dirname, "paws.tsv")
            return {
                "train": train_file_path,
                "dev": dev_file_path,
                "paws": paws_file_path
            }
    def process(self):
        """
            this function turns orign file into form for train in .json like follow
            {
                "text_1":...
                "text_2":...
                "label":...
            }
        """
        file_dict = self.reader()
        
        if self.datasetname == "mnli":        
            if not os.path.exists(os.path.join(self.dirname, "model_feed")):
                print("creat dirs")
                
                writdir = os.path.join(self.dirname, "model_feed")
               
                os.mkdir(writdir)
                for key,value in file_dict.items():
                    print_data = []
                    with open(value, "r", encoding = "utf8") as fr:
                        for item in jsonlines.Reader(fr):
                            print_data.append(
                                {
                                    "text1":item["text1"],
                                    "text2":item["text2"],
                                    "label":item["label"]
                                }
                            )
                        fr.close()  

                    with open(os.path.join(writdir, key + ".json"),"w",encoding="utf8") as fw:
                        fw.write(json.dumps(print_data, indent = 4, ensure_ascii = False) + "\n")
            train_file_path =  os.path.join(self.dirname , "model_feed" , "train.json")
            val_match_file_path = os.path.join(self.dirname , "model_feed" , "val_match.json")
            val_mis_file_path = os.path.join(self.dirname , "model_feed" , "val_mismatch.json")
            test_match_file_path = os.path.join(self.dirname , "model_feed" , "test_match.json")
            test_mis_file_path = os.path.join(self.dirname , "model_feed" , "test_mismatch.json")

            return {
                "tarin" : train_file_path,
                "val_match" : val_match_file_path,
                "val_mismatch" : val_mis_file_path,
                "test_match" : test_match_file_path,
                "test_mismatch" : test_mis_file_path
            }
        if self.datasetname == "hans":
            if not os.path.exists(os.path.join(self.dirname, "model_feed")):
                print("creat dirs")
                
                writdir = os.path.join(self.dirname, "model_feed")
                os.mkdir(writdir)
                for key,value in file_dict.items():
                    print_data = []
                    with open(value, "r", encoding = "utf8") as fr:
                        
                        lines = fr.readlines()[1:]#跳过首行标签行
                        
                        for ele in lines:
                            ele = ele.strip().replace("\n","")
                            
                            ls = ele.split("\t")
                            
                            label = ls[0]
                            text1,text2 = ls[5],ls[6]

                            print_data.append(
                                {
                                    "text1":text1,
                                    "text2":text2,
                                    "label":label
                                }
                            )
                    fr.close()  

                    with open(os.path.join(writdir, key + ".json"),"w",encoding="utf8") as fw:
                        fw.write(json.dumps(print_data, indent = 4, ensure_ascii = False) + "\n")
            train_file_path = os.path.join(self.dirname , "model_feed","train.json")
            test_file_path = os.path.join(self.dirname, "model_feed", "test.json")

            return {
                "train":train_file_path,
                "test":test_file_path
            }
        if self.datasetname == "mnli_hard":
            if not os.path.exists(os.path.join(self.dirname, "model_feed")):
                print("creat dirs")
                
                writdir = os.path.join(self.dirname, "model_feed") 
                os.mkdir(writdir)
                for key, value in file_dict.items():
                    if key == "match_dir":
                        print_data = []
                        with open(os.path.join(value,"multinli_0.9_test_matched_unlabeled_hard.jsonl"),"r",encoding = "utf8") as fr:
                            for item in jsonlines.Reader(fr):
                                print_data.append(
                                    {
                                        "text1":item["sentence1"],
                                        "text2":item["sentence2"],
                                        "pairID":item["pairID"],
                                        "label":None
                                    }
                                )
                            fr.close()

                        with open(os.path.join(value, "multinli_0.9_test_matched_sample_submission_hard.csv"),"r",encoding = "utf8") as fr:
                            reader = csv.reader(fr)
                            result = list(reader)[1:]
                            assert len(result) == len(print_data)
                            for i in range(len(print_data)):
                                assert print_data[i]["pairID"] == result[i][0]
                                print_data[i]["label"] = result[i][1]
                            fr.close()
                        with open(os.path.join(writdir, "hard_test_match.json"),"w",encoding="utf8") as fw:
                            fw.write(json.dumps(print_data, indent = 4, ensure_ascii = False) + "\n")
                    else:
                        print_data = []
                        with open(os.path.join(value,"multinli_0.9_test_mismatched_unlabeled_hard.jsonl"),"r",encoding = "utf8") as fr:
                            for item in jsonlines.Reader(fr):
                                print_data.append(
                                    {
                                        "text1":item["sentence1"],
                                        "text2":item["sentence2"],
                                        "pairID":item["pairID"],
                                        "label":None
                                    }
                                )
                            fr.close()

                        with open(os.path.join(value, "multinli_0.9_test_mismatched_sample_submission_hard.csv"),"r",encoding = "utf8") as fr:
                            reader = csv.reader(fr)
                            result = list(reader)[1:]
                            assert len(result) == len(print_data)
                            for i in range(len(print_data)):
                                assert print_data[i]["pairID"] == result[i][0]
                                print_data[i]["label"] = result[i][1]
                            fr.close()
                        with open(os.path.join(writdir, "hard_test_mismatch.json"),"w",encoding="utf8") as fw:
                            fw.write(json.dumps(print_data, indent = 4, ensure_ascii = False) + "\n")
            match_file_path = os.path.join(writdir , "match.json")
            mismatch_file_path = os.path.join(writdir, "mismatch.json")

            return {
                "match":match_file_path,
                "mismatch":mismatch_file_path
            }
        if self.datasetname == "fever":
            if not os.path.exists(os.path.join(self.dirname, "model_feed")):
                print("creat dirs")
                print(self.dirname)
                writdir = os.path.join(self.dirname , "model_feed")
                os.mkdir(writdir)
            
                for key,value in file_dict.items():
                    print_data = []
                    with open(value, "r", encoding = "utf8") as fr:
                        
                        for item in jsonlines.Reader(fr):
                            
                            if key == "v1" and "v2":
                                print_data.append(
                                {
                                    "text1":item["claim"],
                                    "text2":item["evidence_sentence"],
                                    "label":item["label"]
                                }
                            )
                            else:
                                print_data.append(
                                {
                                    "text1":item["claim"],
                                    "text2":item["evidence"],
                                    "label":item["gold_label"]
                                }
                            )
                            
                            
                        fr.close()  
                    with open(os.path.join(writdir, key + ".json"),"w",encoding="utf8") as fw:
                        fw.write(json.dumps(print_data, indent = 4, ensure_ascii = False) + "\n")
                train_file_path = os.path.join(self.dirname , "model_feed" , "train.json")
                dev_file_path = os.path.join(self.dirname , "model_feed" , "dev.json")
                v1_file_path = os.path.join(self.dirname , "model_feed" , "v1.json")
                v2_file_path = os.path.join(self.dirname , "model_feed" , "v2.json")
                return {
                    "train":train_file_path,
                    "dev":dev_file_path,
                    "v1":v1_file_path,
                    "v2":v2_file_path
                }
        if self.datasetname == "QQP":
            if not os.path.exists(os.path.join(self.dirname, "model_feed")):
                print("creat dirs")
                print(self.dirname)
                writdir = os.path.join(self.dirname , "model_feed")
                os.mkdir(writdir)
           
                for key,value in file_dict.items():
                    print_data = []
                    print(value)
                    with open(value, 'r', encoding='utf-8') as f:
                        next(f)  # 跳过第一行即可
                        if key == "train":
                            for line in f:
                                line = line.strip('\n').split('\t')
                                print_data.append(
                                    {
                                        "text1":line[3],
                                        "text2":line[4],
                                        "label":line[5]#0 and 1
                                    }
                                )
                        if key == "dev":
                            for line in f:
                                line = line.strip('\n').split('\t')
                                print_data.append(
                                    {
                                        "text1":line[3],
                                        "text2":line[4],
                                        "label":line[5]#0 and 1
                                    }
                                )
                        if key == "paws":
                            for line in f:
                                line = line.strip('\n').split("\t")
                                text1 = line[1].strip("b")
                                text1 = text1.strip("'")
                                text1 = text1.strip('"')
                                text2 = line[2].strip("b")
                                text2 = text2.strip("'")
                                text2 = text2.strip('"')
                                print_data.append(
                                    {
                                        "text1":text1,
                                        "text2":text2,
                                        "label":line[3]#0 and 1
                                    }
                                )
                        f.close()  
                    with open(os.path.join(writdir, key + ".json"),"w",encoding="utf8") as fw:
                        fw.write(json.dumps(print_data, indent = 4, ensure_ascii = False) + "\n")
                train_file_path = os.path.join(self.dirname , "model_feed" , "train.json")
                dev_file_path = os.path.join(self.dirname, "model_feed", "dev.json")
                paws_file_path = os.path.join(self.dirname, "model_feed", "paws.json")
                return {
                    "train": train_file_path,
                    "dev": dev_file_path,
                    "paws": paws_file_path
                }
if __name__ == "__main__":
    processor = raw_processer("QQP")

    processor.process()
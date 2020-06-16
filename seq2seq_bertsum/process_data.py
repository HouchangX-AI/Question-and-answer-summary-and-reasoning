import csv
import json
import re

save_path = "PreSumm-master/json_data/"

# with open("train_set.jsonl", "w", encoding="utf-8") as outputfile:
with open("data/AutoMaster_TrainSet.csv", "r") as csv_file:
    formatted_dic = {}
    output_data = []
    p_ct = 0
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print("column names:", ", ".join(row))
            line_count += 1
            continue
        elif len(row) != 6:
            continue
        else:
            brand, model, question, dialogue, report = (
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
            )
            if report == "随时联系":
                continue
            dialogue = " ".join(dialogue.split())

            if "|" in dialogue:
                # splitted_dialogue = [
                #     txt.replace("技师说：", "").replace("车主说：", "")
                #     for txt in dialogue.split("|")
                # ]
                splitted_dialogue = [
                    txt.replace("技师说：", "").replace("车主说：", "")
                    for txt in re.split("，|。|\|", dialogue)
                ]

            elif "技师说" in dialogue:
                splitted_dialogue = [
                    txt.replace("技师说：", "") for txt in re.split("，|。", dialogue)
                ]
            else:
                splitted_dialogue = re.split("，|。", dialogue)
            splitted_dialogue = [
                txt
                for txt in splitted_dialogue
                if "[语音]" not in txt and "[视频]" not in txt and "[图片]" not in txt and txt
            ]
            if len("".join(splitted_dialogue)) < 4:
                continue
            # print(splitted_dialogue, len("".join(splitted_dialogue)))
            # splitted_dialogue = [txt.split("，") for txt in splitted_dialogue]
            processed_text = ["".join([brand, model]), question] + splitted_dialogue
            # print(processed_text)
            processed_text = [list(txt) for txt in processed_text]
            formatted_dic["src"] = processed_text
            formatted_dic["tgt"] = [list(report.replace(" ", ""))]
            output_data.append(formatted_dic.copy())
            if len(output_data) > 16000 - 1:
                pt_file = "{:s}train.{:d}.json".format(save_path, p_ct)
                with open(pt_file, "w") as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(output_data, ensure_ascii=False))
                    p_ct += 1
                    output_data = []

    if len(output_data) > 0:
        print(len(output_data))
        pt_file = "{:s}val.{:d}.json".format(save_path, p_ct)
        with open(pt_file, "w") as save:
            # save.write('\n'.join(dataset))
            save.write(json.dumps(output_data, ensure_ascii=False))
            p_ct += 1
            output_data = []
        line_count += 1

        # outputfile.write(json.dumps(formatted_dic, ensure_ascii=False) + "\n")
        # json.dump(formatted_dic, outputfile, ensure_ascii=False)

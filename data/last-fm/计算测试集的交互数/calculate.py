
f=open("train.txt",encoding='utf-8')
lines=f.readlines()
count=0
for line in lines:
    new_line=line.replace("\n",'').split(' ')
    # print(new_line)
    count=count+len(new_line)
    

# g=open("r.txt",encoding='utf-8')
# lines=g.readlines()
# count=0
# for line in lines:
#     new_line=line.replace("\n",'').replace("]",'').replace("[",'').split(', ')
#     if new_line[0]=="1":
#         count+=1

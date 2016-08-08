dataFile = "/home/mapr/PycharmProjects/Project_OfritSigal/Economic_texts/train_data.txt"
dataFile_fix = "/home/mapr/PycharmProjects/Project_OfritSigal/Economic_texts/train_data-fix.txt"
f = open(dataFile, "rU")

file=[]
sent = ""
for line in f:
    print "line: '" + line + "'"
    if line.endswith('.\n') or line.endswith('!\n') or line.endswith('?\n') or line.endswith(')\n')or line.endswith('.'):
        sent += str(line)
        file.append(sent)
        sent = ""
    else:
        sent += str(line[:-2]) + " "
    print sent

print file
with open(dataFile_fix, 'wb') as nf:
    for line in file:
        print line
        nf.write(line)

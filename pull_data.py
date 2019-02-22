import mysql.connector
from mysql.connector import Error

mySQLconnection = mysql.connector.connect(host='localhost',
                             database='models',
                             user='root',
                             password='')
sql_select_Query = "select * from modelss"
cursor = mySQLconnection .cursor()
cursor.execute(sql_select_Query)
records = cursor.fetchall()
co = 0
print(records)
a = []
a.append(records[0])
a.append(records[1])
a.append(records[2])
for row in a:
    #0 1 2 3 -kaydet 
    print(row[1])
cursor.close()
mySQLconnection.close()
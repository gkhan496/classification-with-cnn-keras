import mysql.connector as conn
import matplotlib.pyplot as plt

class save_database(object):

    def __init__(self):
        self.mydb = conn.connect(
        host="localhost",
        user="root",
        passwd="",
        database="models"
        )

    def create_table(self):
        mycursor = self.mydb.cursor()
        mycursor.execute("CREATE TABLE modelss (id INT AUTO_INCREMENT PRIMARY KEY,conv_1 INTEGER,conv_2 INTEGER,conv_3 INTEGER,last_layer INTEGER);")
    def save_models(self,arr):       
        try:
            mycursor = self.mydb.cursor()
            sql = "INSERT INTO modelss (conv_1,conv_2,conv_3,last_layer) VALUES(%s,%s,%s,%s)"
            mycursor.execute(sql,arr)  
            self.mydb.commit()
        except Exception as e: 
            print(e)
    def pull(self):
        sql_select_Query = "select * from modelss"
        cursor = self.mydb.cursor()
        cursor.execute(sql_select_Query)
        records = cursor.fetchall()
                    
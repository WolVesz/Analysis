# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:58:03 2020

@author: Sean.Holt
"""

import sqlalchemy as sa
import urllib
import pandas as pd

class SQL():
	
	def __init__(self, server, database, username = None, password = None):	
		if username and password:
			params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};" +
			                                 "SERVER={};".format(server) +
			                                 "DATABASE={};".format(database) +
			                                 "UID={};".format(username) +
			                                 "PWD={}".format(password)) 
			engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
			
		else:
			params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};" +
			                                 "SERVER={};".format(server) +
			                                 "DATABASE={};".format(database) +
			                                 "Trusted_Connection=yes")
			engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
		
		self.engine = engine
		self.history = ['Connected to {}:{}'.format(server, database)]
		return
	
	def query(self, sql_query):
		val =  pd.read_sql_query(sql_query, con = self.engine)
		self.history.append(sql_query)
		return val

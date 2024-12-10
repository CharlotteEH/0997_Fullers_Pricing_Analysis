import sqlalchemy as sa
import urllib

params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};"
                                 "SERVER=Neptune;"
                                 "DATABASE=DS_LIVE;"
                                 "Trusted_Connection=yes")

# Connect using the specified parameters
engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
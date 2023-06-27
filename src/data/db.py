from dotenv import load_dotenv
import os
import dbaccess as dbac

load_dotenv()

engine = dbac.create_engine_from_details({
    'drivername': 'postgresql'
    , 'username': os.environ.get('USER_DB')
    , 'password': 'AWS_RDS_IAM_TOKEN'
    , 'host': os.environ.get('ENDPOINT_DB')
    , 'port': os.environ.get('PORT_DB')
    , 'database': os.environ.get('NAME_DB')
    , 'region': os.environ.get('REGION_DB')
    , 'query': {'sslmode': 'require'}
    })

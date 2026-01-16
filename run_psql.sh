sudo systemctl start postgresql
psql -U postgres -h 0.0.0.0 -f helmet_db.sql
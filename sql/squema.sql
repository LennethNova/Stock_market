CREATE TABLE stock_data (
	stock_date DATE NOT NULL,
	open_value NUMERIC NOT NULL,
	high_value NUMERIC NOT NULL,
	low_value NUMERIC NOT NULL,
	close_value NUMERIC NOT NULL,
	adjusting_closing NUMERIC NOT NULL, 
	volume INT NOT NULL,
	stock_name VARCHAR NOT NULL	
);

SELECT * FROM stock_data;
© 2021 GitHub, Inc.
Terms
Privacy
Security
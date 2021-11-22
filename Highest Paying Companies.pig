salary_data  = 
	LOAD 'cleaned_salary_data' 
	USING PigStorage('|') 
	AS (
		timestamp:chararray, 
		company:chararray, 
		level:chararray,
		title:chararray, 
		totalyearlycompensation:int, 
		location:chararray,
		yearsofexperience:float, 
		yearsatcompany:float, 
		tag:chararray,
		basesalary:int, 
		stockgrantvalue:int, 
		bonus:int,
		gender:chararray, 
		otherdetails:chararray, 
		cityid:chararray,
		dmaid:chararray, 
		race:chararray, 
		education:chararray);

-- Calculating the number of employees for each company and the total amount paid to these employees
salary_data_per_company = 
	GROUP salary_data
	BY company;
salary_data_employees_v_payout = 
	FOREACH salary_data_per_company
	GENERATE
		FLATTEN(group) as company, 
		COUNT(salary_data) as count_employees,
		ROUND(AVG(salary_data.totalyearlycompensation)) AS average_avg_salary;


-- Ordering the companies with the highest average salaries
highest_paying_companies = 
	ORDER salary_data_employees_v_payout 
	BY average_avg_salary DESC;

top_10_companies = LIMIT highest_paying_companies 10;
DUMP top_10_companies;
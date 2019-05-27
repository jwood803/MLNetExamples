using Microsoft.ML.Data;

namespace SimpleRegression
{
    public class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience;

        [LoadColumn(1)]
        public float Salary;
    }
}

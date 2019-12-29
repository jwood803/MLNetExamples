using Microsoft.ML.Data;

namespace SimpleRegressionUpdate
{
    public class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience;

        [LoadColumn(1), ColumnName("Label")]
        public float Salary;
    }
}
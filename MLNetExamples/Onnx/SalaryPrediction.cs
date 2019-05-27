using Microsoft.ML.Data;

namespace SimpleRegression
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}

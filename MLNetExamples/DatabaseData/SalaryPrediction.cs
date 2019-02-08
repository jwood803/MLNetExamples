using Microsoft.ML.Data;

namespace DatabaseData
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}

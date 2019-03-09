using Microsoft.ML.Data;

namespace EntityFrameworkData
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}

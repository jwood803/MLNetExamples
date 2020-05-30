using Microsoft.ML.Data;

namespace Ranking
{
    public class RankingData
    {
        [LoadColumn(0), ColumnName("Label")]
        public float Relevance { get; set; }

        [LoadColumn(1)]
        public float GroupId { get; set; }

        [LoadColumn(2, 133)]
        [VectorType(133)]
        public float[] Features { get; set; }
    }
}
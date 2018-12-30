using Microsoft.ML.Runtime.Api;

namespace BinaryClassification
{
    public class TitanicData
    {
        [Column("0")]
        public float PassengerId;

        [Column("1", name: "Label")]
        public bool HasSurvived;

        [Column("2")]
        public float PClass;

        [Column("3")]
        public string Name;

        [Column("4")]
        public string Gender;

        [Column("5")]
        public float Age;

        [Column("6")]
        public float NumOfSiblingsOrSpouses;

        [Column("7")]
        public float NumOfParentOrChildAboard;

        [Column("8")]
        public string Ticket;

        [Column("9")]
        public float Fare;

        [Column("10")]
        public string Cabin;

        [Column("11")]
        public string Embarked;
    }
}

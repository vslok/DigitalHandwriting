using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting
{
    public static class ConstStrings
    {
        public const string CheckText = "lorem ipsum dolor sit amet consectetur adipiscing elit";

        public const string DbName = "DigitalHandwriting.db";

        public const string FirstRegistrationStepDescription = "Come up with a test phrase between 11 and 20 characters long";
        public const string SecondRegistrationStepDescription = "Enter the phrase again.";
        public const string ThirdRegistrationStepDescription = "And one more time.";
        public const string FourthRegistrationStepDescription = "And again.";
        public const string FinalRegistrationStepDescription = "Complete registration";
    }
}

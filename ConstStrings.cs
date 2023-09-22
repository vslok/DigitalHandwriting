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

        public const string FirstRegistrationStepDescription = "For registration input check text. " +
            "Only the current character of the text will be entered. " +
            "After entering the full text, the field will be cleared and the new step will be shown.";
        public const string SecondRegistrationStepDescription = "Enter Check text again";
        public const string ThirdRegistrationStepDescription = "And again";
        public const string FourthRegistrationStepDescription = "Thanks for registration! Click on finalize button for close the window.";
    }
}

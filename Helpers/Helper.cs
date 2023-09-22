using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace DigitalHandwriting.Helpers
{
    public class Helper
    {
        public bool CheckCurrentLetterKeyPressed(KeyEventArgs e, int letterIndex, string upperCaseText)
        {
            KeyConverter kc = new KeyConverter();
            var inputChar = kc.ConvertToString(e.Key);
            if (inputChar == "Space")
            {
                inputChar = " ";
            }
            var currentChar = upperCaseText.ElementAtOrDefault(letterIndex).ToString();

            return inputChar == currentChar;
        }
    }
}

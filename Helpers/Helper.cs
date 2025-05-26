using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace DigitalHandwriting.Helpers
{
    public static class Helper
    {
        private static KeyConverter _keyConverter = new KeyConverter();
        public static bool CheckCurrentLetterKeyPressed(KeyEventArgs e, int letterIndex, string upperCaseText)
        {
            var inputChar = ConvertKeyToString(e.Key);
            var currentChar = upperCaseText.ElementAtOrDefault(letterIndex).ToString();

            return inputChar == currentChar;
        }

        public static string ConvertKeyToString(Key key)
        {
            string keyString = _keyConverter.ConvertToString(key);
            return keyString;
        }
    }
}

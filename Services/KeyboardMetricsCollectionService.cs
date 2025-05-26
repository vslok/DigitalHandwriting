using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using DigitalHandwriting.Context;

namespace DigitalHandwriting.Services
{
    public class KeyboardMetricsCollectionService
    {
        private List<(string KeyName, double Time)> RawKeyDownEvents = new List<(string, double)>();
        private List<(string KeyName, double Time)> RawKeyUpEvents = new List<(string, double)>();

        private readonly List<List<double>> _keyPressedTimes;

        private readonly List<List<double>> _betweenKeysTimes;

        private int _collectingMetricsStep;

        public List<List<double>> KeyPressedTimes {  get { return _keyPressedTimes; } }

        public List<List<double>> BetweenKeysTimes { get { return _betweenKeysTimes; } }

        public KeyboardMetricsCollectionService()
        {
            _collectingMetricsStep = 0;
            _keyPressedTimes = new List<List<double>>(ApplicationConfiguration.RegistrationPassphraseInputs);
            _betweenKeysTimes = new List<List<double>>(ApplicationConfiguration.RegistrationPassphraseInputs);

            for (int i = 0; i < ApplicationConfiguration.RegistrationPassphraseInputs; i++)
            {
                _keyPressedTimes.Add(new List<double>());
                _betweenKeysTimes.Add(new List<double>());
            }
        }

        public void GetCurrentStepValues(
            string testText,
            out List<double> keyPressedValues,
            out List<double> betweenKeysValues)
        {
            if (_collectingMetricsStep < _keyPressedTimes.Count)
            {
                keyPressedValues = _keyPressedTimes[_collectingMetricsStep];
                betweenKeysValues = _betweenKeysTimes[_collectingMetricsStep];
            }
            else
            {
                keyPressedValues = new List<double>();
                betweenKeysValues = new List<double>();
                Trace.WriteLine($"Warning: GetCurrentStepValues called for a step ({_collectingMetricsStep}) that may not have been processed or is out of range.");
            }
        }

        public void OnKeyUpEvent(KeyEventArgs args)
        {
            RawKeyUpEvents.Add((Helper.ConvertKeyToString(args.Key), DateTime.Now.Millisecond / 1000.0));
        }

        public void OnKeyDownEvent(KeyEventArgs args)
        {
            RawKeyDownEvents.Add((Helper.ConvertKeyToString(args.Key), DateTime.Now.Millisecond / 1000.0));
        }

        public void IncreaseMetricsCollectingStep(string testText)
        {
            ConvertRawData(testText);
            _collectingMetricsStep++;
            RawKeyDownEvents.Clear();
            RawKeyUpEvents.Clear();
        }

        private void ConvertRawData(string testText)
        {
            var filteredKeyDownTimes = FilterKeyTimePairsToTheTimeList(this.RawKeyDownEvents, testText);
            var filteredKeyUpTimes = FilterKeyTimePairsToTheTimeList(this.RawKeyUpEvents, testText);

            if (filteredKeyDownTimes.Count != filteredKeyUpTimes.Count)
            {
                Trace.WriteLine($"Critical Error: Filtered KeyDown ({filteredKeyDownTimes.Count}) and KeyUp ({filteredKeyUpTimes.Count}) event counts do not match for testText '{testText}' (Length: {testText.Length}). This indicates inconsistent event recording or filtering for the same text passage.");
                throw new InvalidOperationException("Filtered KeyDown and KeyUp event counts do not match after filtering. Data is inconsistent.");
            }

            if (filteredKeyDownTimes.Count != testText.Length)
            {
                Trace.WriteLine($"Warning: Number of matched key events ({filteredKeyDownTimes.Count}) does not match testText length ({testText.Length}) for text '{testText}'. Some characters may not have corresponding key data. Processing with available data.");
            }

            _keyPressedTimes[_collectingMetricsStep].Clear();
            _betweenKeysTimes[_collectingMetricsStep].Clear();

            for (int i = 0; i < filteredKeyDownTimes.Count; i++)
            {
                _keyPressedTimes[_collectingMetricsStep].Add(filteredKeyUpTimes[i] - filteredKeyDownTimes[i]);

                if (i != 0)
                {
                    _betweenKeysTimes[_collectingMetricsStep].Add(filteredKeyDownTimes[i] - filteredKeyUpTimes[i - 1]);
                }
            }
        }

        public void ResetMetricsCollection()
        {
            _collectingMetricsStep = 0;
            _keyPressedTimes.ForEach(step => step.Clear());
            _betweenKeysTimes.ForEach(step => step.Clear());
            RawKeyDownEvents.Clear();
            RawKeyUpEvents.Clear();
        }

        private List<double> FilterKeyTimePairsToTheTimeList(List<(string KeyName, double Time)> keyEvents, string testText)
        {
            List<double> times = new List<double>();
            var remainingKeyUps = new List<(string KeyName, double Time)>(keyEvents); // Work on a copy

            for (int i = 0; i < testText.Length; i++)
            {
                char targetChar = testText[i];
                (string KeyName, double Time) foundPair = default;
                bool matchFound = false;
                int foundIndex = -1;

                for (int j = 0; j < remainingKeyUps.Count; j++)
                {
                    var currentPair = remainingKeyUps[j];
                    string currentKeyName = currentPair.KeyName;
                    bool isMatch = false;

                    // Check for direct character match (letters, digits, punctuation, symbols)
                    if (currentKeyName.Length == 1 && currentKeyName[0] == targetChar)
                    {
                        isMatch = true;
                    }
                    // Check for special keys by name
                    else if (targetChar == ' ')
                    {
                        isMatch = currentKeyName.Equals("Space", StringComparison.OrdinalIgnoreCase);
                    }
                    else if (targetChar == '\t') // Tab character
                    {
                        isMatch = currentKeyName.Equals("Tab", StringComparison.OrdinalIgnoreCase);
                    }
                    else if (targetChar == '\r' || targetChar == '\n') // Enter/Return characters
                    {
                        // Key.Return (Enter key) is converted to "Enter" by KeyConverter.
                        isMatch = currentKeyName.Equals("Enter", StringComparison.OrdinalIgnoreCase);
                    }
                    // Add more special key mappings here if necessary e.g. Escape, Backspace (if they are expected in testText)

                    if (isMatch)
                    {
                        foundPair = currentPair;
                        foundIndex = j;
                        matchFound = true;
                        break;
                    }
                }

                if (matchFound)
                {
                    times.Add(foundPair.Time);
                    remainingKeyUps.RemoveAt(foundIndex); // Remove the matched pair to ensure sequence
                }
                else
                {
                    // This case means a character in testText did not have a corresponding key-up event
                    // This will likely cause the KeyDownTimes.Count != textKeyUpTimes.Count error later.
                    Trace.WriteLine($"Warning: No matching key-up event found for character '{(targetChar == '\r' ? "\\r" : targetChar == '\n' ? "\\n" : targetChar.ToString())}' (ASCII: {(int)targetChar}) at index {i} in testText '{testText}'.");
                }
            }

            Trace.WriteLine($"[FilterKeyTimePairsToTheTimeList] Times count = {times.Count}, testText length = {testText.Length}");
            return times;
        }
    }
}

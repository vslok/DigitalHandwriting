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
        private List<double> KeyDownTimes = new List<double>();
        private List<(string, double)> KeyUpTimes = new List<(string, double)>();

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
            ConvertRawData(testText);
            keyPressedValues = _keyPressedTimes[_collectingMetricsStep];
            betweenKeysValues = _betweenKeysTimes[_collectingMetricsStep];
        }

        public void OnKeyUpEvent(KeyEventArgs args)
        {
            KeyUpTimes.Add((Helper.ConvertKeyToString(args.Key), DateTime.Now.Millisecond / 1000));
        }

        public void OnKeyDownEvent(KeyEventArgs args)
        {
            KeyDownTimes.Add(DateTime.Now.Millisecond / 1000);
        }

        public void IncreaseMetricsCollectingStep(string testText)
        {

            ConvertRawData(testText);
            _collectingMetricsStep++;
            KeyDownTimes.Clear();
            KeyUpTimes.Clear();
        }

        private void ConvertRawData(string testText)
        {
            var textKeyUpTimes = FilterKeyTimePairsToTheTimeList(KeyUpTimes, testText);
            if (textKeyUpTimes.Count != KeyDownTimes.Count)
            {
                throw new InvalidOperationException("Increase step of metrics collection failed");
            }

            for (int i = 0; i < textKeyUpTimes.Count; i++)
            {
                _keyPressedTimes[_collectingMetricsStep].Add(textKeyUpTimes[i] - KeyDownTimes[i]);

                if (i != 0)
                {
                    _betweenKeysTimes[_collectingMetricsStep].Add(KeyDownTimes[i] - textKeyUpTimes[i - 1]);
                }
            }
        }

        public void ResetMetricsCollection()
        {
            _collectingMetricsStep = 0;
            _betweenKeysTimes.ForEach(step => step.Clear());
            _keyPressedTimes.ForEach(step => step.Clear());
            KeyDownTimes.Clear();
            KeyUpTimes.Clear();
        }

        private List<double> FilterKeyTimePairsToTheTimeList(List<(string, double)> pairs, string pairsText)
        {
            List<double> times = new List<double>();

            for (int i = 0; i < pairsText.Count(); i++)
            {
                var firstPair = pairs.First(pair => pair.Item1[0] == pairsText[i]);
                times.Add(firstPair.Item2);
                pairs.Remove(firstPair);
            }
            Trace.WriteLine($"Times count = {times.Count}, text count = {pairsText.Count()}");
            return times;
        }
    }
}

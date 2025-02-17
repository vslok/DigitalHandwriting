using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace DigitalHandwriting.Services
{
    public class KeyboardMetricsCollectionService
    {
        private List<double> KeyDownTimes = new List<double>();
        private List<(string, double)> KeyUpTimes = new List<(string, double)>();

        private readonly List<List<double>> _keyPressedTimes;

        private readonly List<List<double>> _betweenKeysTimes;

        private readonly List<List<double>> _betweenKeysPressTimes;

        private int _collectingMetricsStep;

        public List<List<double>> KeyPressedTimes {  get { return _keyPressedTimes; } }

        public List<List<double>> BetweenKeysTimes { get { return _betweenKeysTimes; } }

        public List<List<double>> BetweenKeysPressTimes { get { return _betweenKeysPressTimes; } }

        public KeyboardMetricsCollectionService() 
        { 
            _collectingMetricsStep = 0;
            _keyPressedTimes = new List<List<double>>(3);
            _betweenKeysTimes = new List<List<double>>(3);
            _betweenKeysPressTimes = new List<List<double>>(3);

            for (int i = 0; i < 3; i++)
            {
                _keyPressedTimes.Add(new List<double>());
                _betweenKeysTimes.Add(new List<double>());
                _betweenKeysPressTimes.Add(new List<double>());
            }
        }

        public List<double> GetKeyPressedTimesMedians() => Calculations.CalculateMedianValue(_keyPressedTimes);
        public List<double> GetBetweenKeysTimesMedians() => Calculations.CalculateMedianValue(_betweenKeysTimes);

        public List<double> GetBetweenKeysPressTimesMedians() => Calculations.CalculateMedianValue(_betweenKeysPressTimes);

        public List<double> GetKeyPressedTimesDispersions()
        {
            var transposedMatrix = Calculations.MatrixTransposing<double>(_keyPressedTimes);
            return transposedMatrix.Select(val => Calculations.Dispersion(val, Calculations.Expectancy(val))).ToList();
        }

        public List<double> GetBetweenKeysTimesDispersions()
        {
            var transposedMatrix = Calculations.MatrixTransposing<double>(_betweenKeysTimes);
            return transposedMatrix.Select(val => Calculations.Dispersion(val, Calculations.Expectancy(val))).ToList();
        }

        public List<double> GetBetweenKeysPressTimeDispersions()
        {
            var transposedMatrix = Calculations.MatrixTransposing<double>(_betweenKeysPressTimes);
            return transposedMatrix.Select(val => Calculations.Dispersion(val, Calculations.Expectancy(val))).ToList();
        }

        public void GetCurrentStepValues(
            string testText, 
            out List<double> keyPressedValues, 
            out List<double> betweenKeysValues, 
            out List<double> betweenKeysPressValues)
        {
            ConvertRawData(testText);
            keyPressedValues = _keyPressedTimes[_collectingMetricsStep];
            betweenKeysValues = _betweenKeysTimes[_collectingMetricsStep];
            betweenKeysPressValues = _betweenKeysPressTimes[_collectingMetricsStep];
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
                    _betweenKeysPressTimes[_collectingMetricsStep].Add(KeyDownTimes[i] - KeyDownTimes[i - 1]);
                }
            }
        }

        public void ResetMetricsCollection()
        {
            _collectingMetricsStep = 0;
            _betweenKeysTimes.ForEach(step => step.Clear());
            _betweenKeysPressTimes.ForEach(step => step.Clear());
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

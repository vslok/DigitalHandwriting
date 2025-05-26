using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using System.Windows.Markup;
using DigitalHandwriting.Context;

namespace DigitalHandwriting.Converters
{
    class RegistrationInformationStringConverter : MarkupExtension, IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var step = (int)value;

            if (step == ApplicationConfiguration.RegistrationPassphraseInputs + 1)
            {
                return ConstStrings.FinalRegistrationStepDescription;
            }

            return step switch
            {
                0 => ConstStrings.FirstRegistrationStepDescription,
                1 => ConstStrings.SecondRegistrationStepDescription,
                2 => ConstStrings.ThirdRegistrationStepDescription,
                3 => ConstStrings.FourthRegistrationStepDescription,
                _ => (step <= ApplicationConfiguration.RegistrationPassphraseInputs) ? $"Enter passphrase (attempt {step} of {ApplicationConfiguration.RegistrationPassphraseInputs})" : " "
            };
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }

        public override object ProvideValue(IServiceProvider serviceProvider)
        {
            return this;
        }
    }
}

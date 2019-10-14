#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import config

DATA_PATH = config.get_data_path()

MCC_SAVE = DATA_PATH / 'MCCs.csv'
mcc_file = DATA_PATH / 'MCCs_init.csv'

mccs = pd.read_csv(mcc_file)

mccs.fillna('unknown', inplace=True)

mccs['Группа'] = mccs['Группа'].map({"Отели и мотели" : 'hotels',
                                    "Авиалинии, авиакомпании": 'airlines',
                                    'Автомобили и транспортные средств': 'transport_vehicles',
                                    'Аренда автомобилей': 'auto_rental',
                                    'Бизнес услуги': 'business_service',
                                    'Государственные услуги': 'state_service',
                                    'Коммунальные и кабельные услуги': 'utility_service',
                                    'Контрактные услуги': 'contract_service',
                                    'Личные услуги': 'personal_service',
                                    'Магазины одежды': 'clothes', 
                                    'Неизвестно': 'unknown',
                                    'Оптовые поставщики и производители': 'wholesale',
                                    'Поставщик услуг': 'service', 
                                    'Продажи по почте/телефону': 'mail_phone_sales',
                                    'Профессиональные услуги': 'professional_service', 
                                    'Развлечения': 'entertainment',
                                    'Различные магазины': 'other_shops',
                                    'Ремонтные услуги': 'repair_service', 
                                    'Розничные магазины': 'retail_shops',
                                    'Транспорт': 'transport',
                                    'Членские организации': 'membership_organizations'})

mccs.to_csv(MCC_SAVE, index=False)

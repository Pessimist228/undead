"""Промпты стенда шага 0. Список зашит в файл, чтобы прогон можно было повторить.

PROMPTS      50 штук: 20 русских, 20 английских, 4 на код, 6 длинных (2-3 тысячи токенов)
LOAD_PROMPTS 16 штук для фоновой нагрузки, с тестовыми не пересекаются
Длинные промпты собраны арифметикой, без random: результат не зависит от версии Python.
"""

_RU = [
    "Напиши подробный рассказ о том, как дежурный инженер ночью чинит упавший сервер. Не меньше 300 слов.",
    "Объясни школьнику, как работает видеокарта и зачем она нужна нейросетям. Подробно, с примерами.",
    "Составь пошаговую инструкцию, как собрать компьютер из отдельных деталей. Опиши каждый шаг.",
    "Расскажи историю одного дня из жизни оператора станка на картонной фабрике. Не меньше 300 слов.",
    "Сравни три способа хранить резервные копии: на диске дома, в облаке и на ленте. Плюсы и минусы каждого.",
    "Напиши длинное письмо другу о поездке в горы: дорога, погода, люди, еда, обратный путь.",
    "Объясни подробно, что такое очередь запросов и почему сервер может под ней захлебнуться.",
    "Придумай и опиши десять правил хорошего дежурства для команды, которая держит сервис круглые сутки.",
    "Расскажи, как устроен обычный рабочий день в пекарне, от замеса теста до закрытия.",
    "Опиши подробно, как подготовить велосипед к зиме и как на нём ездить в мороз.",
    "Напиши сказку про маленький сервер, который боялся перезагрузки. Не меньше 300 слов.",
    "Объясни, чем оперативная память отличается от диска, и приведи бытовые сравнения.",
    "Составь план недельного похода для четырёх человек: снаряжение, еда, маршрут по дням.",
    "Расскажи, как вырастить помидоры на балконе: от семян до урожая, с типичными ошибками.",
    "Опиши спор двух инженеров о том, стоит ли переписывать старую систему с нуля. Диалогом.",
    "Объясни подробно, почему интернет иногда тормозит вечером и что с этим можно сделать дома.",
    "Напиши подробный обзор вымышленного настольного радиоприёмника: внешний вид, звук, недостатки.",
    "Расскажи, как правильно варить борщ, с объяснением каждого шага и частых ошибок.",
    "Опиши, как прошёл бы первый день человека на новой работе в маленькой мастерской.",
    "Объясни, что такое отказоустойчивость, на примере магазина, в котором одна касса сломалась.",
]

_EN = [
    "Write a detailed story about a night shift operator who notices a failing machine. At least 300 words.",
    "Explain in detail how a hash table works and why lookups are fast. Use everyday comparisons.",
    "Describe step by step how to plan a three-day hiking trip for beginners, including gear and food.",
    "Write a long letter from a lighthouse keeper to a friend in the city about a week of storms.",
    "Compare trains, buses and bicycles as ways to commute. Give advantages and drawbacks of each.",
    "Explain what happens when you type an address into a browser, from the keyboard to the page.",
    "Write a detailed guide to brewing coffee at home with three different methods.",
    "Tell the story of a small repair shop over twenty years, decade by decade.",
    "Explain how a refrigerator works to a curious ten-year-old. Be thorough and patient.",
    "Describe a debate between two engineers about whether to cache everything. Write it as a dialogue.",
    "Write a long product description for an imaginary mechanical keyboard, including its flaws.",
    "Explain why backups fail when they are needed most and how a small team can avoid it.",
    "Describe in detail how bread is made in a small bakery, from the starter to the shelf.",
    "Write a field guide entry for an imaginary bird: appearance, habitat, behaviour, calls.",
    "Explain load balancing with the example of a supermarket with several checkout lanes.",
    "Tell a story about a ship's cook who has to feed a crew for a week with half the supplies.",
    "Describe how to restore an old wooden chair, step by step, with the common mistakes.",
    "Explain what a queue is in computing and what goes wrong when it grows without limit.",
    "Write a detailed travel diary for a slow train journey across a cold country.",
    "Explain how to teach someone to ride a bicycle, including what to do when they are afraid.",
]

_CODE = [
    "Напиши на Python класс очереди с приоритетами на куче, с комментариями к каждому методу и примером.",
    "Write a Python function that parses a log file line by line and counts errors per hour. Explain the code.",
    "Напиши на Python простой HTTP-клиент с повторами и растущей паузой между попытками. Объясни решения.",
    "Write a Python script that walks a directory tree and reports the ten largest files, with comments.",
]

_FACTS_RU = [
    "В три часа ночи на стойке семь отключился блок питания.",
    "Дежурный перевёл нагрузку на соседнюю стойку за две минуты.",
    "Температура в зале поднялась на четыре градуса из-за сломанного кондиционера.",
    "Утром привезли запасной вентилятор, но он оказался другого размера.",
    "Система мониторинга прислала сорок одинаковых предупреждений подряд.",
    "Один из дисков в массиве начал отдавать ошибки чтения.",
    "Инженер заменил диск, не выключая сервер.",
    "После обновления прошивки коммутатор перезагрузился сам.",
    "Резервная линия связи продержала весь трафик почти час.",
    "Журнал доступа показал, что дверь в зал открывали дважды.",
    "На складе нашли три старых сервера без жёстких дисков.",
    "Генератор запустился с третьей попытки.",
    "Топлива в баке оставалось на шесть часов работы.",
    "Заказчик попросил перенести плановые работы на выходные.",
    "Новая стойка не прошла в дверной проём, пришлось снимать колёса.",
    "Ночная смена записала в журнал странный шум из угла зала.",
    "Шум оказался от незакреплённой панели фальшпола.",
    "Батареи бесперебойника выдержали проверку под полной нагрузкой.",
    "Кабель к девятой стойке оказался короче нужного на полметра.",
    "В пятницу вечером отказал один из двух маршрутизаторов.",
    "Запасной маршрутизатор настроили по записям трёхлетней давности.",
    "Счётчик показал, что зал потребляет на десятую часть больше обычного.",
    "Проверка пожарной сигнализации прошла без замечаний.",
    "К концу недели все предупреждения в мониторинге были разобраны.",
]

_FACTS_EN = [
    "At three in the morning the power supply in rack seven shut down.",
    "The operator moved the load to the next rack within two minutes.",
    "The room temperature rose by four degrees because one cooling unit failed.",
    "A spare fan arrived in the morning but it was the wrong size.",
    "The monitoring system sent forty identical warnings in a row.",
    "One disk in the array started returning read errors.",
    "An engineer replaced the disk without powering the server down.",
    "After a firmware update the switch restarted on its own.",
    "The backup link carried all the traffic for almost an hour.",
    "The access log showed that the door to the room was opened twice.",
    "Three old servers without hard drives were found in storage.",
    "The generator started on the third attempt.",
    "The fuel tank held enough for six more hours of work.",
    "The customer asked to move the planned maintenance to the weekend.",
    "The new rack did not fit through the doorway until its wheels were removed.",
    "The night shift wrote down a strange noise from the corner of the room.",
    "The noise came from a loose panel of the raised floor.",
    "The batteries of the power backup passed a test under full load.",
    "The cable to rack nine turned out to be half a metre too short.",
    "On Friday evening one of the two routers failed.",
    "The spare router was configured from notes written three years earlier.",
    "The meter showed the room drawing a tenth more power than usual.",
    "The fire alarm inspection passed without remarks.",
    "By the end of the week every warning in monitoring had been dealt with.",
]


def _long(facts, count, shift):
    return " ".join(facts[(j * 7 + shift) % len(facts)] for j in range(count))


_LONG = [
    "Ниже журнал событий дата-центра.\n\n" + _long(_FACTS_RU, 80, shift)
    + "\n\nПерескажи журнал подробно своими словами и перечисли все неполадки по порядку."
    for shift in (0, 5, 11)
] + [
    "Below is an event log from a data centre.\n\n" + _long(_FACTS_EN, 110, shift)
    + "\n\nRetell the log in detail in your own words and list every fault in order."
    for shift in (2, 9, 17)
]

PROMPTS = _RU + _EN + _CODE + _LONG

LOAD_PROMPTS = [
    "Перечисли двадцать советов тому, кто впервые едет в поход с палаткой, с пояснением к каждому.",
    "Write a long description of a rainy day in a harbour town.",
    "Объясни подробно, как работает лифт и что происходит при отключении электричества.",
    "Explain how tides work and why some coasts have two a day and others one.",
    "Напиши длинный диалог двух соседей, которые спорят о ремонте в подъезде.",
    "Describe the life of a postman in a mountain village across the four seasons.",
    "Расскажи, как выбрать ноутбук для учёбы, и разбери типичные ошибки покупателей.",
    "Write a detailed recipe for vegetable soup with notes on every step.",
    "Опиши подробно, как устроена смена на заводе, от проходной до сдачи станка.",
    "Explain how a library organises its books and how to find one without a catalogue.",
    "Придумай десять вопросов для собеседования механика и напиши к ним хорошие ответы.",
    "Tell a story about two friends who repair an old boat over one summer.",
    "Объясни, почему зимой раньше темнеет, так, чтобы понял ребёнок.",
    "Write a long review of an imaginary board game about running a railway.",
    "Расскажи историю старого трамвая, который ходит по одному маршруту пятьдесят лет.",
    "Explain how to keep a small vegetable garden through a dry summer.",
]

assert len(PROMPTS) == 50 and len(LOAD_PROMPTS) == 16

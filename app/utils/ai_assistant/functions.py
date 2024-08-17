from datetime import datetime, timedelta


async def date_time_filter(date, time):
    date_time = datetime.strptime(f'{date} {time}', '%Y-%m-%d %H:%M')

    print(date_time.strftime('%Y-%m-%d %H:%M'), date_time.weekday())
    if date_time.weekday() in (2, 6):
        return False, "Консультации не проводятся по средам и воскресеньям, давайте выберем другой день :)"
    if not 13 <= date_time.hour <= 18:
        return False, "Консультации проводятся с 13 до 18, давайте выберем другое время :)"
    if date_time.minute not in (0, 30):
        if date_time.minute > 45:
            suggestion_time = date_time.replace(minute=0) + timedelta(hours=1)
        elif date_time.minute < 15:
            suggestion_time = date_time.replace(minute=0) - timedelta(hours=1)
        else:
            suggestion_time = date_time.replace(minute=30)
        return False, f"Могу записать вас на {suggestion_time.strftime('%H:%M')}, подойдёт?"

    return True, f"Записал вас на бесплатную консультацию со специалистом на {date} в {time}"

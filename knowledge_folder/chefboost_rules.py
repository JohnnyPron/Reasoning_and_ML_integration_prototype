def findDecision(obj): #obj[0]: hadUser, obj[1]: hasPersonality, obj[2]: hasGender, obj[3]: hasAge, obj[4]: hadMood, obj[5]: wasWeather, obj[6]: wasTime
	# {"feature": "hadMood", "instances": 70, "metric_value": 3.6673, "depth": 1}
	if obj[4] == 'Sad':
		# {"feature": "hadUser", "instances": 18, "metric_value": 3.0169, "depth": 2}
		if obj[0] == 'Marvin':
			# {"feature": "wasWeather", "instances": 5, "metric_value": 1.9219, "depth": 3}
			if obj[5] == 'Rain':
				return 'Doing_some_exercises'
			elif obj[5] == 'Wind':
				# {"feature": "wasTime", "instances": 2, "metric_value": 1.0, "depth": 4}
				if obj[6] == 'Night':
					return 'Getting_closer'
				elif obj[6] == 'Dinner_time':
					return 'Ordering_some_food'
				else: return 'Ordering_some_food'
			elif obj[5] == 'Sun':
				return 'Telling_a_joke'
			else: return 'Telling_a_joke'
		elif obj[0] == 'Penny':
			# {"feature": "wasWeather", "instances": 2, "metric_value": 1.0, "depth": 3}
			if obj[5] == 'Cloudy':
				return 'Melancholic_music'
			elif obj[5] == 'Wind':
				return 'Verbal_greeting'
			else: return 'Verbal_greeting'
		elif obj[0] == 'Emily':
			return 'Friend_or_family_talk_recommendation'
		elif obj[0] == 'George':
			# {"feature": "wasWeather", "instances": 2, "metric_value": 1.0, "depth": 3}
			if obj[5] == 'Cloudy':
				return 'Telling_a_joke'
			elif obj[5] == 'Storm':
				return 'Classical_music'
			else: return 'Classical_music'
		elif obj[0] == 'Veronica':
			# {"feature": "wasWeather", "instances": 2, "metric_value": 1.0, "depth": 3}
			if obj[5] == 'Sun':
				return 'Reading_a_book'
			elif obj[5] == 'Cloudy':
				return 'Getting_closer'
			else: return 'Getting_closer'
		elif obj[0] == 'John':
			# {"feature": "wasWeather", "instances": 2, "metric_value": 1.0, "depth": 3}
			if obj[5] == 'Rain':
				return 'Melancholic_music'
			elif obj[5] == 'Cloudy':
				return 'Friend_or_family_talk_recommendation'
			else: return 'Friend_or_family_talk_recommendation'
		elif obj[0] == 'Anna':
			return 'Telling_a_joke'
		elif obj[0] == 'Megan':
			return 'Reading_a_book'
		elif obj[0] == 'Eddy':
			return 'Telling_a_joke'
		else: return 'Telling_a_joke'
	elif obj[4] == 'Tired':
		# {"feature": "hasAge", "instances": 12, "metric_value": 1.7296, "depth": 2}
		if obj[3]>16:
			# {"feature": "wasWeather", "instances": 11, "metric_value": 1.4354, "depth": 3}
			if obj[5] == 'Sun':
				return 'Ordering_some_food'
			elif obj[5] == 'Wind':
				return 'Reading_a_book'
			elif obj[5] == 'Cloudy':
				return 'Reading_a_book'
			elif obj[5] == 'Rain':
				return 'Reading_a_book'
			elif obj[5] == 'Snow':
				return 'Hand_wave'
			else: return 'Hand_wave'
		elif obj[3]<=16:
			return 'Rock_music'
		else: return 'Rock_music'
	elif obj[4] == 'Happy':
		# {"feature": "hasAge", "instances": 12, "metric_value": 2.3554, "depth": 2}
		if obj[3]>24:
			# {"feature": "wasTime", "instances": 9, "metric_value": 1.7527, "depth": 3}
			if obj[6] == 'Dinner_time':
				# {"feature": "hadUser", "instances": 2, "metric_value": 1.0, "depth": 4}
				if obj[0] == 'Sam':
					return 'Hand_wave'
				elif obj[0] == 'Apollo':
					return 'Catchy_music'
				else: return 'Catchy_music'
			elif obj[6] == 'Morning':
				# {"feature": "hadUser", "instances": 2, "metric_value": 1.0, "depth": 4}
				if obj[0] == 'Penny':
					return 'Reading_a_book'
				elif obj[0] == 'Emily':
					return 'Verbal_greeting'
				else: return 'Verbal_greeting'
			elif obj[6] == 'Night':
				return 'Hand_wave'
			elif obj[6] == 'Evening':
				return 'Reading_a_book'
			elif obj[6] == 'Afternoon':
				return 'Hand_wave'
			elif obj[6] == 'Noon':
				return 'Reading_a_book'
			else: return 'Reading_a_book'
		elif obj[3]<=24:
			# {"feature": "hadUser", "instances": 3, "metric_value": 1.585, "depth": 3}
			if obj[0] == 'Eddy':
				# {"feature": "wasTime", "instances": 2, "metric_value": 1.0, "depth": 4}
				if obj[6] == 'Morning':
					return 'Verbal_greeting'
				elif obj[6] == 'Afternoon':
					return 'Playing_some_games'
				else: return 'Playing_some_games'
			elif obj[0] == 'John':
				return 'Rock_music'
			else: return 'Rock_music'
		else: return 'Verbal_greeting'
	elif obj[4] == 'Stressed':
		# {"feature": "hasAge", "instances": 10, "metric_value": 2.6464, "depth": 2}
		if obj[3]<=44:
			# {"feature": "hadUser", "instances": 6, "metric_value": 1.7925, "depth": 3}
			if obj[0] == 'Penny':
				return 'Melancholic_music'
			elif obj[0] == 'Emily':
				return 'Going_for_a_walk'
			elif obj[0] == 'Megan':
				return 'Upbeat_music'
			elif obj[0] == 'Luke':
				return 'Rap_music'
			elif obj[0] == 'Marvin':
				return 'Melancholic_music'
			else: return 'Melancholic_music'
		elif obj[3]>44:
			# {"feature": "hadUser", "instances": 4, "metric_value": 1.5, "depth": 3}
			if obj[0] == 'Apollo':
				return 'Friend_or_family_talk_recommendation'
			elif obj[0] == 'Anna':
				return 'Doing_some_exercises'
			elif obj[0] == 'George':
				return 'Classical_music'
			else: return 'Classical_music'
		else: return 'Friend_or_family_talk_recommendation'
	elif obj[4] == 'Depressed':
		# {"feature": "wasTime", "instances": 9, "metric_value": 0.9864, "depth": 2}
		if obj[6] == 'Afternoon':
			return 'Friend_or_family_talk_recommendation'
		elif obj[6] == 'Noon':
			# {"feature": "wasWeather", "instances": 2, "metric_value": 1.0, "depth": 3}
			if obj[5] == 'Sun':
				return 'Friend_or_family_talk_recommendation'
			elif obj[5] == 'Snow':
				return 'Melancholic_music'
			else: return 'Melancholic_music'
		elif obj[6] == 'Evening':
			return 'Friend_or_family_talk_recommendation'
		elif obj[6] == 'Night':
			return 'Staying_quiet'
		elif obj[6] == 'Dinner_time':
			return 'Friend_or_family_talk_recommendation'
		else: return 'Friend_or_family_talk_recommendation'
	elif obj[4] == 'Bored':
		# {"feature": "wasTime", "instances": 6, "metric_value": 1.4591, "depth": 2}
		if obj[6] == 'Morning':
			return 'Doing_some_exercises'
		elif obj[6] == 'Evening':
			# {"feature": "hadUser", "instances": 2, "metric_value": 1.0, "depth": 3}
			if obj[0] == 'Penny':
				return 'Going_for_a_walk'
			elif obj[0] == 'John':
				return 'Playing_some_games'
			else: return 'Playing_some_games'
		elif obj[6] == 'Noon':
			return 'Going_for_a_walk'
		else: return 'Going_for_a_walk'
	elif obj[4] == 'Angry':
		# {"feature": "hadUser", "instances": 3, "metric_value": 1.585, "depth": 2}
		if obj[0] == 'Eddy':
			return 'Rap_music'
		elif obj[0] == 'Apollo':
			return 'Melancholic_music'
		elif obj[0] == 'John':
			return 'Friend_or_family_talk_recommendation'
		else: return 'Friend_or_family_talk_recommendation'
	else: return 'Friend_or_family_talk_recommendation'

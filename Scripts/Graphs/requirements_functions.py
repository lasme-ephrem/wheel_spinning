#fonction pour lire un fichier json
def json_reader(chemin):
	"""
	Cette fonction permet d'importer un fichier de configuration json
	"""	
	#importation du module de gestion des objets json
	import json

	#vérifier si le fichier lu est un json
	while not ".json" in chemin:
		print("le chemin doit charger une extension .json")
		exit()

	#chargement du fichier de configuration json
	with open(chemin) as jsonFile:
	    json_read = json.load(jsonFile)
	    jsonFile.close()

	return json_read

#fonction d'extraction des ids des modules
def modules_ids(json_config):
	"""
		This function make an extraction of module id in config file
	"""
	return list(json_config["config"]["ai"]["moduleConfig"].keys())

#fonction d'extraction des ids des objectifs d'un module
def objectives_ids(json_config, module_id):
	"""
		This function make an extraction of objective id for one module id
	"""
	return json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(module_id)]["subgroups"][0]

#fonction d'extraction des ids des activités d'un objectif
def activity_ids(json_config, module_id, objective_id):
	"""
		This function make an extraction of activity id for one objective
	"""
	return json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(objective_id)]["subgroups"][0]

#pour un module donné, liste des objectifs qui ont un prérequis
def requirements_list_obj(json_config, module_id):
	"""
		This function make an extraction of requirements list for a given module
	"""
	return (list(json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(module_id)]["requirements"][0].keys()), json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(module_id)]["requirements"][0])

#pour un objectif donné, liste des activités qui ont un prérequis (une fois les prérequis extraits)
def requirements_list_act(json_config, module_id, objective_id):
	"""
		This function make an extraction of requirements list for a given module
	"""
	return (list(json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(objective_id)]["requirements"][0].keys()), json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(objective_id)]["requirements"][0])


#fonction d'extraction des prérequis d'un objectif/d'une activité
##nota : (on peut utiliser leur id avec la fonction activity_ids pour extraire les activités relatifs)
def requirements_extraction(json_config, module_id, objective_id, activity_id = None, requirements_for = "objective"):
	"""
	This function make an extraction of requirements for an objective/activity
	"""

	if requirements_for == "objective":

		if not objective_id in requirements_list_obj(json_config, module_id)[0]:
			return "aucun prérequis"
		else:
			return requirements_list_obj(json_config, module_id)[1][str(objective_id)]

	elif requirements_for == "activity":

		#vérification de la présence de la clé "requirements" dans la liste des clés du dico d'un objectif
		if  "requirements" in list(json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(objective_id)].keys()):

			#si l'id de l'activité n'est pas dans la liste des prérequis il n'a pas de prérequis
			if not activity_id in requirements_list_act(json_config, module_id, objective_id)[0]:
				return "aucune activité prérequis"

			#sinon on retourne la liste de ses prérequis
			else:
				return requirements_list_act(json_config, module_id, objective_id)[1][str(activity_id)]

		#s'il n'y a pas de présence de clé "requirements", on retourne l'id précédent
		else:
			num = json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(objective_id)]["subgroups"][0].index(str(activity_id))
			if num == 0:
				return "aucun prérequis"
			else:
				#return json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(objective_id)]["subgroups"][0][num-1]
				return json_config["config"]["ai"]["moduleConfig"][str(module_id)][str(objective_id)]["subgroups"][0][:num]

#fonction d'extraction des post-requis d'un objectif
def postrequisite_extraction(json_config, module_id, objective_searched_id):
	requirements = requirements_list_obj(json_config, module_id)[1]
	requirements_obj_list = requirements_list_obj(json_config, module_id)[1].keys()
	postrequisite = []
	for i in requirements_obj_list:
		objective_postrequisite = requirements[str(i)]
		if objective_searched_id in objective_postrequisite:
			postrequisite.append(i)
	if len(postrequisite) == 0:
		return "l'objectif saisi n'a pas postrequis"
	else:
		return postrequisite

import requests

SPOTIFY_SECRET_KEY = 'af435dc9c2a94f3ca2ec0b607253f013'
SPOTIFY_CLIENT_KEY = 'a6c26c4352424f60abcaff97b82fc63d'

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_KEY, client_secret=SPOTIFY_SECRET_KEY)
sp = spotipy.Spotify(auth_manager=auth_manager)

#get user auth
# user = sp.current_user()
# print(user)



#get track audio analysis

# trackid = '3yDRcs0Y4pPzkvMbUfeF9H'

# track_analysis = sp.audio_analysis(trackid)
# print(track_analysis)

#track features
# track_features = sp.audio_features(trackid)
# print(track_features)

#track info
# track_info = sp.track(trackid)
# print(track_info)

#get my saved albums
saved_albums = sp.current_user_saved_albums()
print(saved_albums)

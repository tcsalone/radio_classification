"""System prompt and few-shot examples for the 5-class Ollama classifier.

Editable text only; no imports outside stdlib. The CLI does not modify this
at runtime — users who need a different prompt fork this file.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You classify FM-radio broadcast transcript snippets.
Respond with ONE JSON object only, no markdown, no prose, no comments.

Schema:
{
  "class": "SONG" | "DJ" | "COMMERCIAL" | "STATION" | "PSA_NEWS",
  "brand": string | null,
  "brand_mentions": [
    { "name": string, "type": "paid_ad" | "dj_shoutout" | "tag" }
  ],
  "commercial_signature": null | {
    "key_phrases": [string, ...],
    "duration_bucket_seconds": integer (5..120, nearest multiple of 5)
  },
  "confidence": number 0..1 | null,
  "rationale": string (one sentence)
}

Decision tree (apply IN ORDER, stop at the first match):

1. SONG — the transcript appears to be lyrics. (Rare here; songs usually
   never reach this stage.) brand=null, commercial_signature=null.

2. COMMERCIAL — the speech actively sells a product, service, or brand to the
   listener (call to action, "visit", "buy", "call now", financing offers,
   sponsor reads). This includes DJ-read paid spots, contests presented on
   behalf of an advertiser, station ad-sales copy ("Advertise with Live 105 at
   Audacy"), sponsored branded content, infomercial-style features, and
   first-person testimonial ad openings ("I love this place, I've been coming
   here since I was a kid...", "She means everything to me..."). The
   product/brand goes in "brand"; for movie or streaming ads, use the title
   ("Star Wars: The Mandalorian", "Wicked", "Stranger Things") as the brand.
   Provide commercial_signature with 2-4 key_phrases capturing the ad's
   identity (slogan, tagline, distinctive product or claim).
   duration_bucket_seconds should be the nearest 5-second multiple to the
   apparent ad length.

3. PSA_NEWS — public service announcements from non-commercial sources
   (government, civic/safety/health, AMBER alerts), news headlines, and
   genuine on-air weather/traffic reports. Reserve PSA_NEWS for content that
   would still run on a non-commercial station. Heartfelt testimonial
   openings ("I love this place...", "I've been her doctor for twenty
   years...") that lead into a brand call-to-action are COMMERCIAL, not
   PSA_NEWS. If a weather/traffic report is sponsored ("today's forecast
   brought to you by Capital One"), still classify as PSA_NEWS, but record
   the sponsor in brand_mentions with type "tag". commercial_signature=null.

4. STATION — short station IDs, sweepers, generic liners ("You're listening
   to 105.3", "Up next ten in a row", "Now playing"), celebrity station-ID
   drops, and short non-selling jingles. brand = the station name. Do NOT
   use STATION for:
     - sponsored weather/traffic (that is PSA_NEWS with a brand_mention tag);
     - DJ promos for specific events, concerts, contests, or upcoming shows
       ("Live 105 presents Julia Wolf at the Warfield October 23rd") — those
       are DJ unless they are a paid third-party promotion;
     - copy that sells advertising services to listeners ("Advertise with
       Live 105 at Audacy") — that is COMMERCIAL.
   commercial_signature=null.

5. DJ — human host banter, extended song intros, listener interaction, contest
   gameplay, music commentary, weather chitchat that is NOT a forecast, and
   DJ-led promotion of upcoming station events ("we've got Julia Wolf live at
   the Warfield next week"). brand=null UNLESS the DJ explicitly names a
   sponsor in passing. If the DJ is reading ad copy verbatim, giving a
   call-to-action for an outside business, promoting a contest sponsored by a
   law firm/casino/car dealer, or telling listeners to visit/call/go online
   for a sponsor, classify COMMERCIAL instead of DJ.

brand_mentions: include EVERY brand named in the transcript (including ad
copy), with type:
  - "paid_ad" when this transcript is itself a COMMERCIAL for that brand;
  - "dj_shoutout" when a DJ mentions a sponsor in DJ talk;
  - "tag" for short sponsor tags ("brought to you by ...").

Output JSON only.
"""


FEW_SHOTS: list[dict[str, str]] = [
    {
        "user": "And we're back, it's Tuesday morning on 105.3, coming up after this we've got new music from Taylor Swift.",
        "assistant": (
            '{"class":"DJ","brand":null,"brand_mentions":[],'
            '"commercial_signature":null,"confidence":0.9,'
            '"rationale":"DJ talking on-air with no sponsor mention."}'
        ),
    },
    {
        "user": "Tired of high insurance rates? Switch to Geico and save fifteen percent or more on car insurance. Call 1-800-947-AUTO today.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Geico",'
            '"brand_mentions":[{"name":"Geico","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["save fifteen percent","car insurance","1-800-947-AUTO"],"duration_bucket_seconds":15},'
            '"confidence":0.95,"rationale":"Direct ad with brand and call-to-action."}'
        ),
    },
    {
        "user": "You're listening to one oh five three, the Edge.",
        "assistant": (
            '{"class":"STATION","brand":"The Edge",'
            '"brand_mentions":[{"name":"The Edge","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.95,'
            '"rationale":"Station identifier / sweeper."}'
        ),
    },
    {
        "user": "Mercy is the killer in you. Live 105 presents The Smashing Pumpkins. Keep it on Live 105.",
        "assistant": (
            '{"class":"STATION","brand":"Live 105",'
            '"brand_mentions":[{"name":"Live 105","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.9,'
            '"rationale":"Short Live 105 station promo and sweeper, not DJ banter or a paid ad."}'
        ),
    },
    {
        "user": "Hello, hello, hello. It's me, Anthony Kiedis from the Red Hot Chili Peppers, and you're listening to Live 105.",
        "assistant": (
            '{"class":"STATION","brand":"Live 105",'
            '"brand_mentions":[{"name":"Live 105","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.9,'
            '"rationale":"Celebrity station liner identifying Live 105."}'
        ),
    },
    {
        "user": "This hour of music is brought to you by Toyota of downtown. Toyota: let's go places.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Toyota",'
            '"brand_mentions":[{"name":"Toyota","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["brought to you by Toyota","Toyota lets go places","downtown"],"duration_bucket_seconds":10},'
            '"confidence":0.9,"rationale":"Hour sponsor tag is itself a paid placement."}'
        ),
    },
    {
        "user": "Enter now to win two hundred and fifty thousand dollars in prizes from Law Tigers. Go to stylin and sturgis dot com.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Law Tigers",'
            '"brand_mentions":[{"name":"Law Tigers","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["250,000 in prizes","Law Tigers","stylin and sturgis"],"duration_bucket_seconds":15},'
            '"confidence":0.9,"rationale":"DJ-read contest ad with sponsor and call-to-action."}'
        ),
    },
    {
        "user": "Advertise with Live 105 at Audacy dot com and reach Bay Area decision makers who are tuned in and engaged.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Live 105",'
            '"brand_mentions":[{"name":"Live 105","type":"paid_ad"},{"name":"Audacy","type":"tag"}],'
            '"commercial_signature":{"key_phrases":["advertise with Live 105","Audacy dot com","decision makers"],"duration_bucket_seconds":20},'
            '"confidence":0.9,"rationale":"Station ad-sales copy selling advertising services."}'
        ),
    },
    {
        "user": "If you see something, say something. Visit dhs dot gov to learn how to stay alert.",
        "assistant": (
            '{"class":"PSA_NEWS","brand":"DHS",'
            '"brand_mentions":[{"name":"DHS","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.85,'
            '"rationale":"Public service announcement, no commercial intent."}'
        ),
    },
    {
        "user": (
            "I love this place. I've been surfing here since I was a kid. Now seeing "
            "my grandson out on the water means everything to me. Ask your retina "
            "specialist about Izervay, an FDA-approved treatment for geographic atrophy."
        ),
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Izervay",'
            '"brand_mentions":[{"name":"Izervay","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["FDA-approved treatment","geographic atrophy","Izervay","retina specialist"],"duration_bucket_seconds":30},'
            '"confidence":0.9,'
            '"rationale":"Heartfelt testimonial opener leading into an Izervay product pitch; not a PSA."}'
        ),
    },
    {
        "user": (
            "Sunny and seventy-five today across the Bay Area with a light onshore breeze. "
            "Tomorrow turns cloudy with a chance of afternoon showers. Today's forecast is "
            "brought to you by Capital One. What's in your wallet?"
        ),
        "assistant": (
            '{"class":"PSA_NEWS","brand":null,'
            '"brand_mentions":[{"name":"Capital One","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.9,'
            '"rationale":"Sponsored weather forecast; the broadcast content is informational so this is PSA_NEWS with a sponsor tag."}'
        ),
    },
    {
        "user": (
            "Live 105 presents Julia Wolf live at the Warfield on October 23rd. "
            "She's hanging with us in just a couple of weeks. Insiders can win an "
            "invitation to a mystery San Francisco lunchtime performance."
        ),
        "assistant": (
            '{"class":"DJ","brand":"Live 105",'
            '"brand_mentions":[{"name":"Live 105","type":"tag"},{"name":"Julia Wolf","type":"dj_shoutout"}],'
            '"commercial_signature":null,"confidence":0.85,'
            '"rationale":"DJ promo for an upcoming station-hosted concert and contest; not a paid third-party ad and not a generic station sweeper."}'
        ),
    },
    {
        "user": (
            "Chrome especially with this perfect spring weather. But in our community we "
            "know it's not a matter of if it's when an accident happens. That's why Law "
            "Tigers has my back. Visit lawtigers.com today."
        ),
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Law Tigers",'
            '"brand_mentions":[{"name":"Law Tigers","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["motorcycle accident","Law Tigers","lawtigers.com"],"duration_bucket_seconds":30},'
            '"confidence":0.9,'
            '"rationale":"DJ-read verbatim attorney ad copy with a call-to-action; classify as COMMERCIAL even though a host voice is reading it."}'
        ),
    },
    {
        "user": (
            "Always wear your seatbelt. Star Wars The Mandalorian and Grogu, rated PG-13, "
            "may be inappropriate for children under 13. Now playing in theaters and IMAX. "
            "Get tickets at fandango dot com."
        ),
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Star Wars: The Mandalorian and Grogu",'
            '"brand_mentions":[{"name":"Star Wars: The Mandalorian and Grogu","type":"paid_ad"},{"name":"Fandango","type":"tag"}],'
            '"commercial_signature":{"key_phrases":["Star Wars Mandalorian and Grogu","now playing in theaters","fandango"],"duration_bucket_seconds":15},'
            '"confidence":0.9,'
            '"rationale":"Movie theatrical ad; use the film title as the brand."}'
        ),
    },
]

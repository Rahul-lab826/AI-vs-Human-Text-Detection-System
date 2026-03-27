"""
Dataset generator for AI vs Human Text Detection.
Produces exactly 22,500 samples:
  7,500 genuine human text        (label = 0)
  7,500 AI formal text            (label = 1)
  7,500 AI humanized text         (label = 1)

KEY DESIGN PRINCIPLE:
  Human samples: sentence length JUMPS, topic drifts, self-contradictions,
                 incomplete thoughts, thinking-out-loud, emotional spikes,
                 intentional grammar breaks, personal specifics, time refs.
  AI formal:     uniform 10-15 word sentences, discourse markers, zero
                 emotional variance, no contractions, passive/third person.
  AI humanized:  surface slang/typos but deep uniformity — no topic drift,
                 no real contradictions, logical flow maintained, slang
                 evenly distributed not randomly clustered.
"""
import csv
import random
import re
import os

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN BASE SAMPLES  (label = 0)
# Every sample must have at least 3 of:
#   - sentence length jump (short sentence next to long one)
#   - mid-text topic change
#   - self-contradiction
#   - incomplete thought (trailing...)
#   - thinking out loud (wait, ok so, actually)
#   - emotional spike (caps, !!, ??)
#   - intentional grammar mistake
#   - physical sensation or location reference
#   - time reference (rn, yesterday, last week, this morning)
#   - personal specific detail (name, place, number)
# ─────────────────────────────────────────────────────────────────────────────

# ── Type 1: Normal human typing without slang ─────────────────────────────────
# Short simple sentences, personal details, no formal structure, imperfect flow
human_base_normal = [
    "I was at the store today and forgot to get milk. Typical me honestly. The weather has been pretty weird this week. Not sure what to make of it. My sister called last night which was nice.",
    "Went for a walk this morning. It was cold but not too bad. Saw a dog by the park. Made me miss having a dog. Might look into that at some point.",
    "Had a really long day at work. Nothing went wrong exactly, just a lot of small things. Meetings that ran over. That kind of stuff. Looking forward to the weekend.",
    "Made dinner tonight for the first time in a while. Nothing fancy. Just pasta. It came out okay. Better than ordering out again I guess.",
    "Finished that book I've been reading. It was good. Not great, but good. Would probably recommend it if someone asked. The ending felt a bit rushed.",
    "Tried to fix the leaky faucet myself. Watched two videos about it. Still not fixed. Called a plumber. This is probably the right call.",
    "Been pretty tired lately. Not sure if it's the weather or just life stuff. Going to try going to bed earlier this week. Probably won't stick to it.",
    "My coworker brought in snacks today. Made the afternoon much better. Small things like that make a difference honestly. We should do it more often.",
    "Went to the dentist today. Not my favorite thing. Everything was fine though. Just a checkup. Should probably go more regularly.",
    "Got a package in the mail I forgot I ordered. It was a phone case. The original one finally cracked. This new one is fine. Not exciting but it works.",
    "Had coffee with an old friend today. We hadn't seen each other in maybe a year. It was nice catching up. Time flies. Should do it more often.",
    "The commute was particularly bad this morning. Train delays. Nothing I could do about it. Got in about 20 minutes late. My manager didn't mention it.",
    "Cleaned the bathroom today. Not fun but needed to happen. The shower especially. Feel better now that it's done. Should have done it a week ago.",
    "Tried a new sandwich place near my office. It was decent. A bit pricey for what it was. Probably won't go back regularly but not bad.",
    "My landlord finally fixed the heating. Only took three weeks. The apartment is actually warm now. This is a significant improvement.",
    "Finished a project at work that's been dragging on. Good to have it done. Not sure what's next. Something will come up.",
    "Ran into my neighbor in the elevator. We talked about the building for a minute. Nice enough person. I never remember his name though.",
    "Watched a movie last night. It was fine. Not great, not terrible. Fell asleep in the last twenty minutes. Looked up the ending this morning.",
    "Got a lot done today actually. Made a list in the morning and went through most of it. That almost never happens. Decent day overall.",
    "My phone battery is terrible lately. Barely lasts half the day. Probably time to get a new one. Been putting it off.",
    "Cooked for the first time this week. Everything else has been leftovers or takeout. It wasn't complicated but it felt good. Should cook more.",
    "Had a headache most of the afternoon. Not debilitating, just annoying. Took some ibuprofen. It helped eventually. Fine now.",
    "Walked to work today instead of taking the train. Took about 30 minutes. Nice enough morning for it. Good to get some air.",
    "My sister texted asking for advice about something. We talked for a while. I'm not sure my advice was great but it seemed helpful.",
    "Tried to get to bed early last night. Made it to around 11. That's earlier than usual. Felt noticeably better this morning because of it.",
]

# ── Simple short human samples (label = 0) — no slang, no features required ──
human_base_simple = [
    "My name is Rahul.",
    "I went to class today.",
    "I finished my assignment.",
    "The weather is nice today.",
    "I had lunch with a friend.",
    "I woke up early this morning.",
    "My name is Alex and I study biology.",
    "I went to the library after school.",
    "The meeting was at 10am.",
    "I bought groceries on the way home.",
    "It rained today.",
    "I called my mom last night.",
    "I have a test tomorrow.",
    "I forgot to charge my phone.",
    "My dog needs a walk.",
    "I made coffee this morning.",
    "I read for an hour before bed.",
    "The bus was late today.",
    "I sent the email already.",
    "I need to do laundry.",
    "I had cereal for breakfast.",
    "My sister is visiting next week.",
    "I finished work early today.",
    "The internet was slow all morning.",
    "I ran into an old classmate.",
    "I took a long shower.",
    "My back hurts a little.",
    "I watched the news this morning.",
    "I forgot to water my plant.",
    "I have a dentist appointment Friday.",
    "I ordered pizza for dinner.",
    "I cleaned my desk.",
    "I got a haircut today.",
    "I forgot my umbrella.",
    "My roommate is cooking.",
    "I missed the bus.",
    "I got to work on time.",
    "I paid my rent.",
    "I have a headache.",
    "I went to bed late last night.",
    "I skipped breakfast.",
    "The coffee is still hot.",
    "I replied to the message.",
    "I need a new charger.",
    "I finished my homework.",
    "I took notes during class.",
    "I brought lunch from home.",
    "I found my keys.",
    "I forgot my password again.",
    "I set an alarm for tomorrow.",
    "The package arrived today.",
    "I have a lot of emails to read.",
    "I took a nap after work.",
    "I need to call the doctor.",
    "I finally finished that report.",
    "I made a to-do list.",
    "I went for a short walk.",
    "I ate leftovers for lunch.",
    "I fixed the issue finally.",
    "I dropped my pen.",
    "I wore a jacket today.",
    "I studied for two hours.",
    "I forgot to reply.",
    "I refilled my water bottle.",
    "I finished reading that chapter.",
    "I switched off the light.",
    "I printed the document.",
    "I updated my notes.",
    "I packed my bag the night before.",
    "I arrived five minutes early.",
    "I submitted the form online.",
    "I checked the weather before leaving.",
    "I paid for it in cash.",
    "I waited in line for twenty minutes.",
    "I had soup for dinner.",
    "I charged my laptop.",
    "I returned the library book.",
    "I texted him back.",
    "I washed the dishes.",
    "I watered the plants.",
    "I took my medicine.",
    "I watched the game last night.",
    "I missed the first few minutes.",
    "I put on a jacket.",
    "I chose the window seat.",
    "I left work at five.",
    "I parked near the entrance.",
    "I locked the door.",
    "I turned down the heat.",
    "I wrote in my journal.",
    "I listened to music on the way.",
    "I downloaded the app.",
    "I reset my password.",
    "I updated the spreadsheet.",
    "I booked the appointment.",
    "I confirmed the reservation.",
    "I organized my files.",
    "I replied to the feedback.",
    "I joined the meeting a bit late.",
    "I shared the document.",
    "I submitted the assignment an hour before the deadline.",
    "I reviewed my notes before the exam.",
    "I asked a question at the end of the lecture.",
    "I took a screenshot and sent it.",
    "I made a list before going shopping.",
    "I rescheduled the call for tomorrow.",
    "I packed my lunch the evening before.",
    "I remembered to bring my ID.",
    "I double-checked the address before leaving.",
    "I thanked the cashier.",
    "I held the door open for someone.",
    "I returned the change.",
    "I sat by the window.",
    "I ordered the same thing I always get.",
    "I left a tip.",
    "I asked for the check.",
    "I waited for the light to change.",
    "I took the stairs instead.",
    "I got off at the wrong stop.",
    "I transferred lines at the station.",
    "I checked the map twice.",
    "I bought a bottle of water.",
    "I picked up a free newspaper.",
    "I sat in a different spot than usual.",
    "I ran a little to catch the train.",
    "I got a seat near the door.",
    "I stood the whole ride.",
    "I got to class just in time.",
    "I sat in the front row.",
    "I borrowed a pen from the person next to me.",
    "I copied down the key points.",
    "I checked my phone after class.",
    "I ate alone today.",
    "I found a quiet table.",
    "I left before the rush.",
    "I heated up my food.",
    "I forgot to bring a fork.",
    "I ate at my desk.",
    "I grabbed something quick.",
    "I had a snack around three.",
    "I drank water instead of coffee.",
    "I made tea instead.",
    "I skipped the meeting.",
    "I forgot to mute myself.",
    "I lost track of the time.",
    "I checked the agenda before joining.",
    "I took notes on my laptop.",
    "I spoke up once during the call.",
    "I had trouble connecting at first.",
    "I left the camera off.",
    "I rejoined after getting disconnected.",
    "I sent a follow-up email after.",
    "I read the instructions twice.",
    "I filled out the form.",
    "I signed the document.",
    "I made a copy just in case.",
    "I filed it in the wrong folder first.",
    "I moved it to the right place.",
    "I deleted the duplicate.",
    "I renamed the file.",
    "I saved it before closing.",
    "I backed it up.",
    "I exported the data.",
    "I shared the link.",
    "I left a comment.",
    "I replied to the thread.",
    "I tagged the right person.",
    "I marked it as done.",
    "I moved the task to this week.",
    "I checked everything off the list.",
    "I postponed one thing.",
    "I delegated that part.",
    "I added a note for tomorrow.",
    "I set a reminder.",
    "I closed the tabs I didn't need.",
    "I turned off notifications for a bit.",
    "I put my phone face down.",
    "I focused for about an hour.",
    "I took a short break.",
    "I stretched at my desk.",
    "I refilled my coffee.",
    "I opened a window.",
    "I lowered the brightness.",
    "I switched to dark mode.",
    "I plugged in my headphones.",
    "I put on some background music.",
    "I muted the TV.",
    "I turned down the volume.",
    "I paused and came back to it.",
    "I finished the last item on the list.",
    "I logged off for the day.",
    "I packed up my things.",
    "I turned off the monitor.",
    "I grabbed my bag.",
    "I said goodbye on the way out.",
    "I took the long way home.",
    "I stopped at the store.",
    "I picked up milk.",
    "I got home a little after six.",
    "I sat down for a few minutes before doing anything.",
    "I changed out of my work clothes.",
    "I started dinner earlier than usual.",
    "I set the table.",
    "I watched something while I ate.",
    "I did the dishes right after.",
    "I checked my phone before bed.",
    "I read a little.",
    "I fell asleep faster than expected.",
    "I woke up once in the middle of the night.",
    "I went back to sleep without much trouble.",
    "I got up when my alarm went off.",
    "I made the bed.",
    "I brushed my teeth.",
    "I got dressed.",
    "I had breakfast.",
    "I left on time.",
]

# ── Type 2: Human technical/explaining style ───────────────────────────────────
# Thinking out loud, uncertainty phrases, self-doubt visible, incomplete explanations
human_base_technical = [
    "so the way i understand it the cpu basically processes instructions one at a time right. or maybe multiple but like in sequence. i think. honestly i might be wrong about this but thats how i always thought about it. anyway the point is it handles calculations",
    "okay so recursion is when a function calls itself i think. the base case is when it stops? i always mix up which part is which. but basically it keeps going until it hits that condition. at least that's my understanding. could be off on the details.",
    "from what i can tell the database stores everything in tables and the query just kind of asks for specific rows. like filtering i guess. i'm probably oversimplifying this. the joins are the part i never fully got.",
    "so git basically tracks changes to your files over time i think. commits are like snapshots. branches let you work on different versions. i think that's right. the rebasing part always confused me honestly.",
    "the way i think about APIs is like ordering at a restaurant. you ask for something specific and it comes back with what you asked for. or an error. i know that's a simplification. but it helps me think about it.",
    "honestly i'm not entirely sure how https works. something with encryption and certificates i think. the browser checks if the site is who it says it is. i know there's more to it but that's the gist as far as i understand.",
    "so async programming is about not waiting around for things right. like if something takes a long time you move on and come back to it. i think. callbacks or promises or something. i always have to look this up.",
    "the way i understand machine learning is the model looks at a lot of examples and figures out patterns. i know it's more complicated than that. gradient descent and all that. but basically it learns from data. right?",
    "so containers are like lightweight virtual machines i think. except not really. they share the operating system but are isolated from each other. docker is the main one i've heard of. i'm probably missing something.",
    "from what i gather, caching is just storing things temporarily so you don't have to recalculate them every time. like saving the answer. makes things faster. the tricky part is knowing when to clear the cache i think.",
    "okay so i've been trying to understand what exactly a compiler does. it takes your code and turns it into something the computer can actually run i think. there's an intermediate step. bytecode or something. i'd have to look it up.",
    "the way routing works in web apps i think is it maps urls to specific functions. so when you go to a certain address it runs the right code. i know there's more to it but that's my basic understanding of it.",
    "so version control is for keeping track of changes and being able to go back if something breaks. i think the main benefit is collaboration. multiple people working on the same code without overwriting each other. mostly.",
    "authentication versus authorization. one is proving who you are and the other is what you're allowed to do i think. i always mix up which is which. tokens are involved somehow. jwt or session or something.",
    "so the stack is where local variables go and the heap is for dynamically allocated memory i think. or the other way around. one of them is faster. this is the kind of thing i look up every time.",
]

# ── Type 3: Human emotional without slang ────────────────────────────────────
# Real emotional sequence, short frustrated sentences, personal timeline, no formal connectors
human_base_emotional = [
    "I had the worst day. Nothing went right from the morning. I spilled coffee, missed the bus, and then got to work late. My manager noticed. I just wanted to go home. At least the day is over now I guess. Tomorrow will be better.",
    "Really struggling today. Not sure why exactly. Everything feels harder than it should. The smallest tasks feel overwhelming. I know it will pass. Just not right now.",
    "Got some bad news this afternoon. Nothing catastrophic but still not good. Needed a few minutes to process it. Feeling a bit better now. Just one of those days.",
    "Honestly pretty frustrated right now. I did everything I was supposed to do and it still didn't work out. That's frustrating. Not much else to say about it.",
    "Had a good day today. Actually a really good day. Things went well at work and I got home at a decent time. Made a nice dinner. Small things but they add up.",
    "Feeling pretty low today for no obvious reason. Just one of those days where everything feels a bit grey. Not looking for solutions, just acknowledging it. It happens.",
    "Something nice happened today and it put me in a good mood for the whole afternoon. It was small. Someone just said something kind. But it mattered.",
    "Really worried about something and I can't stop thinking about it. I know worrying doesn't help. I know that. Still doing it though.",
    "Today was fine. Not great, not bad. Just a normal day. I feel like I should appreciate those more. The days that are just quiet and unremarkable.",
    "Cried a little today over nothing in particular. Or nothing specific. Just felt necessary. Feel better now. Strange how that works.",
    "Got some really good news today. Like genuinely good. The kind you weren't expecting. Still processing it a little. In a good way.",
    "Had an argument with someone I care about. Not a big one but it stayed with me all day. We're fine now. But I keep replaying it.",
    "Feeling tired in a way that sleep doesn't fix. That particular kind of tired. Going to take it easy tonight. Just need some quiet.",
    "Really proud of myself for something today. It was small and no one noticed but I noticed. That feels like it counts.",
    "Anxious about something coming up next week. The anticipation is always worse than the thing itself. I know this. Doesn't stop the feeling though.",
    "Honestly relieved. Something I was worried about resolved itself and I feel about 10 pounds lighter. Good outcome. Moving on.",
    "Disappointed today. Not devastated, just disappointed. The kind where you take a breath and readjust and keep going. That kind.",
    "Missing someone today for no particular reason. They're fine, I just thought about them. Sent a quick message. Sometimes that's enough.",
    "Something went better than expected today and it caught me off guard. I was prepared for it to go badly. Pleasant surprise.",
    "Nervous about a conversation I need to have. Rehearsed it in my head a few times. Will probably be fine. Still nervous though.",
    "Felt genuinely happy today. Not performed happiness or telling myself to be grateful. Just actually felt it. Nice when that happens.",
    "Made a mistake today and I'm being harder on myself about it than I should be. It was fixable. I fixed it. Still annoyed though.",
    "Feeling grateful today for no specific reason. Just aware of what I have. Those moments are worth noting.",
    "Today felt long in a good way. Lots happened. By the end of it I felt like I'd actually done something. That's a good feeling.",
    "Burned out a little today. Pushed through it. Got things done. But running on empty by the end. Need a proper rest.",
]

human_base = [
    # Short sentence + long sentence jumps, time refs, topic drift, personal details
    "so i was literally about to sleep when i remembered i forgot to submit that form. ugh. anyway its fine probably? idk. my roomate (terrible speller btw lol) said it doesnt matter but shes also the one who thought australia was in europe so. ok going to bed",
    "CANNOT believe i just sat through a 2 hour meeting that could have been an email. my hand hurts from fake-taking notes. also i havent eaten since this morning and i am SPIRALING a little. anyway. hi.",
    "ok so i made this amazing pasta yesterday?? or i thought i did. ate the leftovers today and it was actually kind of terrible. idk what happened. maybe i remembered it wrong. or maybe hunger just makes everything taste better. probably the second one.",
    "my dog bit me today. just lightly. but still. she's never done that before and i don't know what to do about it. she seemed sorry afterward which sounds insane to say about a dog but she did that thing where she puts her chin on your knee and just stares. forgiven i guess.",
    "started a new book! loved the first 40 pages. then got distracted, put it down for 3 days, picked it back up and had absolutely no idea what was happening. who is derek. why is derek on a boat. i don't know derek.",
    "real talk i have been in a MOOD all week and i cannot figure out why. like nothing bad happened. everything is technically fine. but my brain is doing that thing where it just manufactures anxiety out of thin air. fun times!! anyway how are you",
    "ok wait i forgot to tell you. so last tuesday (i think it was tuesday) my coworker mike brought his dog to the office and the dog found my lunch bag. like, found it, opened the zipper, and ate half my sandwich. mike bought me a new lunch but still. the zipper part gets me.",
    "i said something really dumb in my job interview and i've been replaying it for approximately 6 days straight. the interviewer laughed but like... was it an uncomfortable laugh?? or a real laugh?? i will never know and it will haunt me forever",
    "tried meditating for the first time. lasted 4 minutes before i started making a grocery list in my head. then i started thinking about whether i left the stove on. then i got up to check (i didn't leave the stove on). so that was that.",
    "UPDATE: i fixed it. well. kind of fixed it. it works now but i'm not sure why it works, which means it could stop working at any time for the same reason i don't understand. but for now. it works. celebrate the small wins i guess",
    "i love my job i love my job i love my job. ok that was a lie. i like my job. sometimes. not today. today was a lot. but it's fine. i'm fine. ask me again friday",
    "went to the gym this morning for the first time in like 6 weeks. felt incredible. walked out feeling like a new person. sat down when i got home and then basically didn't move for 4 hours. so. mixed results.",
    "my phone just autocorrected 'on my way' to 'on my war' and i hit send before i saw it. my boss is probably very confused right now and i cannot explain this fast enough",
    "spent $47 at target. went in for paper towels. came out with paper towels and also a candle, two things i didn't know i needed, one thing i definitely didn't need, and somehow no paper towels because i forgot to grab them.",
    "wait ok so this is embarrassing. i've been mispronouncing my coworker's name for three months. THREE MONTHS. her name is elena not elaina. she corrected me gently today in the elevator. just the two of us. nowhere to go.",
    "my apartment is SO clean right now. which means i was procrastinating on something important. i cleaned the oven. i haven't cleaned the oven in maybe a year. whatever i was procrastinating on must have been really bad.",
    "the funniest thing just happened. or. wait, actually no it wasn't funny, it was kind of upsetting, but now that it's over it's a little bit funny. ok so basically my presentation slides were the wrong version and i presented the wrong data. to 30 people. for 20 minutes.",
    "i keep telling myself i'll go to sleep early tonight. it is currently 1:17am. i have to be up at 6:30. i'm looking at videos of baby goats. this is fine.",
    "SO frustrated rn. asked for one simple thing from IT. just one. they sent me a link to a FAQ that doesn't answer my question. i replied asking again. they sent a different FAQ. it also doesn't answer my question. this has been going on for 4 days.",
    "genuinely cannot decide if i want to move cities or not. part of me wants a fresh start. part of me really likes my apartment and the coffee place on the corner knows my order. that coffee place might be the only thing keeping me here actually. no that's an exaggeration. maybe.",
    "woke up at 3am last night for absolutely no reason. brain immediately started thinking about something embarrassing from 2014. just selected it from the archive and played it on full volume. went back to sleep at 5. alarm went off at 6:30. great.",
    "my mom called me to ask how to forward an email. we stayed on the phone for two hours. i did not teach her to forward emails. we talked about my aunt's divorce and a bird she's been watching in the backyard. the email is probably still not forwarded.",
    "i lost my keys this morning. found them. lost my wallet. found it in the keys place. lost my headphones. never found them. left 20 minutes late anyway and made it to work exactly on time which somehow felt like a personal victory.",
    "the confidence i feel after a good meeting cannot be destroyed. update: it can be destroyed. one bad email and it's gone. took about 45 minutes.",
    "ok i need to be honest. i've watched the same show 4 times. not because it's that good. it's just... comfortable? like i don't have to pay attention. i know what happens. my brain can just rest. is that sad? i think that might be a little sad.",
    "just got back from a run!! first one in months. 2.3 miles. i had to stop and walk twice. my neighbor saw me struggling up the small hill by the park and gave me a thumbs up which somehow made it worse.",
    "i'm pretty sure my plant is dying. it's been dying for 6 months though so maybe that's just its vibe now. we're both just existing. getting through it. i respect that.",
    "had a dream i got fired and woke up genuinely stressed for like 30 minutes before i remembered i literally got promoted last month. my brain is wild. truly working against me at all times.",
    "the audacity of this alarm clock. i set it for 7:30. it went off at 7:30. i turned it off. i went back to sleep. i woke up at 10:15. this happens every single saturday. every one.",
    "just realized i've been saying 'irregardless' for years. like confidently. in professional settings. someone let me do this. no one stopped me. friends don't let friends use made-up words and yet.",
    "grocery shopping when hungry should be illegal. $130 later. i have chips, three kinds of cheese, a rotisserie chicken, snacks i've never tried before, and zero actual meals. cannot explain the cheese. didn't need three kinds.",
    "my roommate and i watched a movie last night and had completely different opinions on it. like. were we watching the same thing?? she thought the ending was beautiful. i thought it made no sense. we are still friends but i'm thinking about it.",
    "email inbox: 847 unread. not going to deal with it today. probably not tomorrow either. at some point this becomes an archaelogy project and i'm just going to have to start fresh. this is my plan. i'm at peace with it.",
    "nap math: planned 20 minutes, slept 2 hours, now it's dark outside, i don't know what day it feels like, my back hurts, and i somehow feel more tired than before i lay down. classic.",
    "tried a new recipe for dinner. followed every single step. still somehow managed to burn the onions and undercook the rice at the same time. the laws of physics were not on my side tonight. ordered pizza.",
    "the social anxiety of calling to make a doctor's appointment is so real. i've been needing to make this appointment for two months. i've picked up the phone 6 times. i'll do it tomorrow. or actually maybe the day after.",
    "my best friend from high school (sarah) texted me out of nowhere today and it was like no time had passed at all. then i looked at the clock and realized we'd been texting for 3 hours and i had skipped a meeting. worth it though.",
    "this is going to sound dramatic but i think my coworker is slightly afraid of me and i have no idea what i did. she was totally normal two weeks ago. now she does this thing where she gives very short answers and doesn't make eye contact. did i do something?? i don't think i did anything.",
    "update on the plant situation: i looked up what i was doing wrong and i've been watering it too much. turns out i've been killing it with kindness. too much water. all this time i thought i wasn't doing enough and i was doing TOO much. there's probably a metaphor here.",
    "today i learned that the thing i've been doing at work for 8 months is not actually the right process. there's a whole other way to do it that takes literally half the time. no one told me. i did it wrong for 8 months. i am fine.",
    "i told myself i would stop impulse buying things online. i bought a lamp at 11pm last night. it arrived today. it's a nice lamp. i have no regrets but i also have no justification.",
    "my friend group chat has been silent for 3 weeks. then someone sent a meme. then it was silent again for 2 more days. then suddenly 47 messages in 20 minutes and everyone is making plans for saturday. this is how it always goes and i love it.",
    "can't stop thinking about what i should have said in that argument from last week. i think of the perfect comeback at approximately 2am. by morning i don't care anymore. but in the moment of 2am it feels very important.",
    "ok so the meeting i was dreading? it was actually fine. better than fine. they liked my idea. i've been anxious about this for a week and a half and it took 12 minutes and went totally fine. i should remember this next time but i won't.",
    "every single time i try to be productive on a weekend i end up doing exactly one (1) productive thing and then treating myself for the rest of the day as a reward. today i responded to one email. i have been rewarding myself for 4 hours.",
    "just got off a phone call i completely forgot about. they called, i panicked, pretended i was ready, it went fine. i have no idea what we agreed to. i'm going to have to piece it together from the calendar invite and pray.",
    "i'm an adult who cried at a dog food commercial today. the dog was running through a field to meet its owner. it was very touching. i was not prepared. the commercial is 30 seconds long.",
    "sat in my car in the parking lot for 20 minutes after getting home because the song wasn't over yet. had to see it through. sometimes you just have to see it through.",
    "the number of things on my to-do list that have been on my to-do list for 6+ months is a form of psychological warfare against myself that i am actively losing.",
    "my apartment building neighbor and i have been waving to each other for 2 years and i just realized i don't know his name and it's too late to ask. we're committed to the wave now. forever.",
    # Short punchy + contradictions
    "i'm definitely a morning person. (i am not a morning person.)",
    "this week was so good. wait no it wasn't. it was actually kind of rough. but it ended okay. so: mixed.",
    "i said i'd work out today. i worked out today!! wait no i didn't. i thought about working out, made the playlist, and then... yeah.",
    "everything is fine. update: one thing is not fine. update: it resolved itself. update: a different thing is now not fine. ok.",
    "i love cooking. i hate doing the dishes. these two facts are in direct conflict with my life choices.",
    "ngl this morning was rough. but the coffee helped. the coffee always helps. until it makes me anxious and then the anxiety doesn't help.",
    "feeling very productive today. update from 4pm: was not productive. did 20% of one thing. called it.",
    "ok. deep breath. this is manageable. wait is it manageable? i think it's manageable. actually i'm not sure. ok deep breath.",
]

# Human augmentation parts — designed to ADD the required authentic features
_h_short_bursts = [
    "ugh.", "nope.", "anyway.", "wait.", "ok so.", "never mind.", "actually—",
    "help.", "wow.", "hm.", "oh.", "right.", "yeah no.", "lol ok.", "k.",
    "??", "wHAT.", "ANYWAY.", "...yeah.", "moving on.",
]
_h_long_sentences = [
    "like i genuinely cannot believe how fast this week went and also simultaneously how slowly, it feels like monday was both yesterday and three weeks ago and i don't know how time works anymore",
    "the thing nobody tells you about being an adult is that you spend a truly significant portion of your life waiting — waiting for appointments, waiting for replies, waiting to feel ready for things you're never actually going to feel ready for",
    "i keep thinking about this conversation i had with my dad like four years ago where he said something really offhand and kind of wise and i didn't really absorb it at the time but it just like. hits different now",
    "you know that feeling where you're not sad exactly but you're not happy either and it's not really neutral it's more like your brain is in a weird standby mode and you're just kind of going through the motions until something snaps you out of it",
    "i went to return something at the store and the whole process took 45 minutes because the system was down and the manager had to do it manually and there was a whole thing and by the end i was invested in whether linda's supervisor was going to be nice about it",
]
_h_topic_drifts = [
    "anyway completely unrelated but i need a haircut so bad. like my hair is doing things.",
    "speaking of which i haven't talked to jamie in a month and i should probably fix that.",
    "also unrelated: found a 20 dollar bill in a jacket pocket. peak human experience.",
    "ok changing topics: does anyone else's fridge make a weird sound at night or is that just mine.",
    "oh also i forgot to mention — the parking situation near my office is a NIGHTMARE now.",
    "btw totally different thing — my neighbor got a puppy and i saw it this morning. tiny. chaotic.",
]
_h_contradictions = [
    "i'm over it. wait no i'm not over it.",
    "actually that was fine. no wait it wasn't fine. ok it was kind of fine.",
    "i said i wasn't going to care about this and i am clearly caring about this.",
    "honestly i'm not even stressed about it anymore. (i am still stressed about it.)",
    "i meant what i said. i'm taking it back. ok i still meant it but i'm less sure now.",
    "i don't need their approval. (i would still like their approval. i'm working on this.)",
]
_h_trailing = [
    "and then i just kind of...",
    "not sure where i was going with that tbh",
    "anyway it doesn't matter i guess it's just...",
    "idk it's fine. or maybe not. whatever.",
    "but yeah. i don't know. it's a lot.",
]
_h_thinking_aloud = [
    "wait actually no —", "ok so thinking about this more —",
    "actually hold on —", "no wait let me rephrase —",
    "ok so ACTUALLY —", "hmm. ok. so. —", "wait no ok —",
]
_h_emotional_spikes = [
    "I LITERALLY CANNOT.", "WHAT IS HAPPENING.", "OK NO THIS IS TOO MUCH.",
    "WHY IS THIS SO HARD???", "UNACCEPTABLE honestly.", "???? explain.",
    "i'm NOT okay.", "this is SO annoying omg.", "absolutely NOT.",
]
_h_personal_details = [
    "so my friend alex (she's a nurse in chicago) said —",
    "i live on the 4th floor and the elevator has been broken for 2 weeks —",
    "my coworker dan, who is 43 and has worked here for 11 years, said —",
    "i have exactly $47 in my checking account until friday —",
    "the coffee place i go to is on 3rd and it charges $6.50 for an oat latte —",
    "my car (2017, 90k miles, making a concerning sound) —",
]
_h_time_refs = [
    "literally rn", "yesterday though", "last tuesday honestly",
    "this morning at like 7am", "two weeks ago and i'm still thinking about it",
    "at 2am last night", "three days ago and i'm still mad",
    "earlier today", "like five minutes ago", "this weekend",
]
_h_physical = [
    "my back is absolutely killing me rn",
    "i can smell my neighbor's cooking from the hallway",
    "it is so cold in this office i'm wearing a coat at my desk",
    "my head hurts from staring at this screen",
    "my hands are freezing i need to find the thermostat",
]


def _inject_authentic_features(base_text):
    """Add random authentic features to a human text sample."""
    features_added = 0
    result = base_text

    roll = random.random()

    # Feature 1: Short sentence jump (always inject if long base)
    if len(base_text.split()) > 20 and random.random() < 0.7:
        burst = random.choice(_h_short_bursts)
        result = burst + " " + result
        features_added += 1

    # Feature 2: Topic drift
    if random.random() < 0.45:
        result = result.rstrip(". ") + ". " + random.choice(_h_topic_drifts)
        features_added += 1

    # Feature 3: Contradiction
    if random.random() < 0.35:
        result = result.rstrip(". ") + ". " + random.choice(_h_contradictions)
        features_added += 1

    # Feature 4: Thinking aloud (mid-text interruption)
    if random.random() < 0.30:
        words = result.split()
        if len(words) > 8:
            mid = len(words) // 2 + random.randint(-2, 2)
            interrup = random.choice(_h_thinking_aloud)
            words.insert(mid, interrup)
            result = " ".join(words)
            features_added += 1

    # Feature 5: Emotional spike
    if random.random() < 0.30:
        result = result.rstrip(". ") + ". " + random.choice(_h_emotional_spikes)
        features_added += 1

    # Feature 6: Trailing thought
    if random.random() < 0.25:
        result = result.rstrip(". ") + " " + random.choice(_h_trailing)
        features_added += 1

    # Feature 7: Personal specific detail (inject early)
    if random.random() < 0.35:
        detail = random.choice(_h_personal_details)
        result = detail + " " + result
        features_added += 1

    return result


def make_human():
    roll = random.random()

    # Combined pool: original slangy + normal + technical + emotional + simple
    _all_human = (human_base + human_base_normal +
                  human_base_technical + human_base_emotional +
                  human_base_simple)

    if roll < 0.20:
        # Simple short human text — no augmentation required
        return random.choice(human_base_simple)

    elif roll < 0.35:
        # Normal/technical/emotional base — returned plain (no augmentation)
        pool = human_base_normal + human_base_technical + human_base_emotional
        return random.choice(pool)

    elif roll < 0.48:
        # Plain base from any pool, no augmentation
        return random.choice(_all_human)

    elif roll < 0.60:
        # Long base + authentic features injected
        base = random.choice(human_base)
        return _inject_authentic_features(base)

    elif roll < 0.70:
        # Short burst + long ramble + topic drift
        short = random.choice(_h_short_bursts)
        long_s = random.choice(_h_long_sentences)
        drift = random.choice(_h_topic_drifts)
        time_r = random.choice(_h_time_refs)
        return f"{short} {long_s}. ({time_r}.) {drift}"

    elif roll < 0.80:
        # Thinking aloud + contradiction + trailing
        base = random.choice(human_base)
        think = random.choice(_h_thinking_aloud)
        contra = random.choice(_h_contradictions)
        trail = random.choice(_h_trailing)
        words = base.split()
        mid = max(1, len(words) // 2)
        words.insert(mid, think)
        return " ".join(words) + ". " + contra + " " + trail

    elif roll < 0.88:
        # Personal detail + physical sensation + time ref + base
        personal = random.choice(_h_personal_details)
        physical = random.choice(_h_physical)
        time_r = random.choice(_h_time_refs)
        base = random.choice(human_base)
        return f"{personal} {physical} ({time_r}). {base}"

    elif roll < 0.94:
        # Emotional spike + base + contradiction
        spike = random.choice(_h_emotional_spikes)
        base = random.choice(human_base)
        contra = random.choice(_h_contradictions)
        return f"{spike} {base} {contra}"

    else:
        # Two bases joined with drift + topic change
        b1 = random.choice(human_base)
        b2 = random.choice(human_base)
        drift = random.choice(_h_topic_drifts)
        return b1.rstrip(".") + ". " + drift + ". anyway — " + b2


# ─────────────────────────────────────────────────────────────────────────────
# AI FORMAL BASE SAMPLES  (label = 1)
# Requirements: consistent 10-15 word sentences, discourse markers, zero
# emotional variance, passive/third person, perfect grammar, no contractions
# ─────────────────────────────────────────────────────────────────────────────

ai_formal_base = [
    "Artificial intelligence represents a significant advancement in computational science. It enables machines to perform tasks requiring human intelligence. The implications span numerous domains of modern life. Continued research is essential to realizing its full potential.",
    "The implementation of machine learning has demonstrated potential across diverse industries. Applications in healthcare, finance, and transportation have been widely documented. Each domain presents distinct challenges and opportunities for optimization. Evidence-based evaluation of outcomes remains critical.",
    "Climate change presents one of the most significant challenges of the contemporary era. Comprehensive responses from governments and organizations are required. Coordinated action across sectors is broadly considered necessary. The evidence supporting urgent intervention is well established.",
    "Effective communication in professional environments necessitates clarity and precision. Understanding of audience expectations is equally essential. Contextual norms shape what constitutes appropriate professional discourse. These factors are well documented in organizational communication research.",
    "The global economy operates as an interconnected system of markets and institutions. Regulatory frameworks facilitate the exchange of goods, services, and capital. Disruptions in one sector frequently propagate across interconnected systems. Systemic analysis is required to understand these dynamics.",
    "Advancements in renewable energy technology have substantially reduced costs. Solar and wind generation have seen the most notable price reductions. These trends have significant implications for energy policy and planning. The trajectory of adoption is expected to continue.",
    "Research in neuroscience has revealed the remarkable plasticity of the human brain. The brain demonstrates capacity to reorganize in response to experience. This finding has significant implications for education and rehabilitation. The evidence base for neuroplasticity is well established.",
    "The management of organizational change requires careful attention to communication. Stakeholder engagement is a key component of successful transitions. The psychological dimensions of human adaptation must also be addressed. These factors are consistently associated with positive change outcomes.",
    "Sustainable agriculture seeks to meet food production needs responsibly. The capacity of future generations must not be compromised. Evidence-based practices are essential for long-term viability. Policymakers and practitioners must coordinate their efforts accordingly.",
    "The philosophical tradition of ethics provides frameworks for moral reasoning. Consequentialist, deontological, and virtue-based approaches have each been influential. Each framework offers distinct analytical advantages and limitations. A comprehensive ethical analysis typically draws on multiple traditions.",
    "Digital transformation in education integrates technology into pedagogical methods. Curriculum design and institutional administration are both affected. Evidence for technology's impact on learning outcomes is mixed. Context-specific evaluation of implementation is therefore recommended.",
    "Public health interventions require evidence-based approaches. Epidemiological data must inform program design and evaluation. Behavioral and sociocultural factors are also relevant considerations. Effective interventions address multiple determinants simultaneously.",
    "Quantum computing leverages principles of quantum mechanics for computation. Superposition and entanglement are foundational to its operation. These properties enable computations beyond classical capabilities. Practical applications are currently in early stages of development.",
    "The conservation of biodiversity is essential for maintaining ecosystem services. Pollination, water purification, and climate regulation all depend on biodiversity. Loss of species represents an irreversible reduction in ecological resilience. International coordination is required to address this challenge.",
    "Social media platforms have fundamentally altered patterns of communication. Information dissemination occurs at unprecedented scale and speed. The implications for public discourse and democratic participation are significant. Empirical research on these effects continues to develop.",
    "Urban planning encompasses a multidisciplinary approach to city design. Functionality, sustainability, and quality of life are primary objectives. Effective planning integrates input from diverse stakeholder groups. The outcomes depend on sustained commitment to evidence-based practice.",
    "The historical development of democracy spans several key periods. Ancient Athenian governance established foundational principles. Enlightenment philosophy extended and refined these concepts. Contemporary democratic systems reflect this accumulated historical development.",
    "Advances in genomic sequencing have transformed biological research. Genetic variation, disease susceptibility, and evolutionary history can now be examined. These insights have direct implications for medicine and public health. The pace of advancement continues to accelerate.",
    "The relationship between economic inequality and social mobility is complex. Structural factors and institutional arrangements both play important roles. Individual circumstances also contribute to observed outcomes. Comprehensive policy responses must address each of these dimensions.",
    "The study of linguistics encompasses multiple subfields of inquiry. Phonology, morphology, syntax, semantics, and pragmatics each address distinct phenomena. Together they provide a comprehensive account of language structure and use. The field continues to generate significant empirical findings.",
    "Effective time management involves setting clear and measurable priorities. Breaking tasks into manageable components improves completion rates. Minimizing context-switching between projects preserves cognitive resources. Regular review of progress supports sustained performance.",
    "The stages of software development typically include requirements analysis and design. Implementation, testing, and deployment follow in sequence. Ongoing maintenance ensures continued functionality after release. Iterative methodologies revisit these phases in shorter cycles.",
    "It is important to note that correlation does not imply causation. Causal inference requires temporal precedence and elimination of confounders. The conditions for causal claims are often not met in observational studies. Practitioners should interpret such findings with appropriate caution.",
    "The evidence suggests that early intervention yields better outcomes. This pattern has been documented across developmental and educational domains. Public health applications have similarly benefited from early action. The consistency of these findings strengthens the case for intervention.",
    "Investment in preventive measures yields higher long-term returns. Equivalent expenditure on remedial interventions consistently shows lower returns. The data supporting this conclusion are robust and well replicated. This finding has direct implications for resource allocation decisions.",
    "Productivity can be improved through the consistent application of evidence-based strategies. Research supports the use of structured routines and goal-setting. Monitoring progress provides useful feedback for continuous improvement. These techniques are applicable across a range of professional contexts.",
    "Sleep is an essential component of overall health and well-being. Adequate rest significantly improves cognitive performance and emotional regulation. The consequences of chronic sleep restriction are well documented in the literature. Prioritizing sleep is therefore a rational and evidence-based choice.",
    "Exercise has been shown to have positive effects on mental health. Reductions in symptoms of depression and anxiety have been documented. These benefits appear across diverse populations and age groups. Physical activity is therefore an important component of mental health care.",
    "Budgeting is an important financial skill for achieving monetary goals. Systematic planning enables tracking of income and expenditure over time. Regular review of spending patterns supports informed financial decisions. These practices are associated with improved long-term financial outcomes.",
    "Mindfulness practices are associated with reductions in psychological stress. Meditation and breathing exercises have been studied in controlled trials. Improvements in emotional well-being have been documented across multiple populations. These practices are increasingly integrated into clinical settings.",
]

# AI formal augmentation
_af_sentence_templates = [
    "The role of {topic} in {domain} has been extensively examined in the literature.",
    "Evidence consistently supports the importance of {topic} for {domain}.",
    "{Topic} represents a foundational element of contemporary {domain}.",
    "The implications of {topic} for {domain} are significant and well documented.",
    "Practitioners and researchers agree that {topic} warrants ongoing attention in {domain}.",
]
_af_topics = [
    "artificial intelligence", "machine learning", "data governance",
    "climate policy", "public health", "urban sustainability",
    "financial regulation", "digital literacy", "supply chain resilience",
    "renewable energy", "data privacy", "organizational leadership",
    "behavioral economics", "computational biology", "natural language processing",
    "cybersecurity", "democratic accountability", "environmental policy",
    "healthcare innovation", "cognitive science", "social cohesion",
]
_af_domains = [
    "modern institutional practice", "evidence-based policy development",
    "organizational effectiveness", "long-term sustainability",
    "cross-disciplinary scholarship", "global systems governance",
    "scientific understanding", "professional decision-making",
    "public administration", "contemporary research methodology",
]
_af_connectors = [
    "Furthermore, ", "Additionally, ", "In this context, ",
    "As a result, ", "Consequently, ", "In particular, ",
    "Building on this, ", "More broadly, ", "Importantly, ",
    "Moreover, ", "It should also be noted that ",
]
_af_closers = [
    "This conclusion is supported by the preponderance of available evidence.",
    "Further empirical work will refine our understanding of these dynamics.",
    "Policymakers would benefit from incorporating these insights.",
    "The practical implications of this finding are considerable.",
    "Ongoing evaluation and adaptation remain essential components of effective practice.",
    "The evidence base for this position continues to expand substantially.",
    "This perspective has gained increasing acceptance among practitioners.",
    "The data are consistent with this interpretation across multiple contexts.",
]


def make_ai_formal():
    roll = random.random()

    if roll < 0.30:
        # Template sentence + connector + formal closer
        topic = random.choice(_af_topics)
        domain = random.choice(_af_domains)
        tmpl = random.choice(_af_sentence_templates)
        s1 = tmpl.format(topic=topic, domain=domain, Topic=topic.capitalize())
        conn = random.choice(_af_connectors)
        closer = random.choice(_af_closers)
        return f"{s1} {conn}{closer[0].lower()}{closer[1:]}"

    elif roll < 0.55:
        # Base + formal closer
        base = random.choice(ai_formal_base)
        return base + " " + random.choice(_af_closers)

    elif roll < 0.72:
        # Connector + base
        base = random.choice(ai_formal_base)
        conn = random.choice(_af_connectors)
        return conn + base[0].lower() + base[1:]

    elif roll < 0.86:
        # Two template sentences joined
        topic1 = random.choice(_af_topics)
        topic2 = random.choice(_af_topics)
        domain1 = random.choice(_af_domains)
        domain2 = random.choice(_af_domains)
        s1 = random.choice(_af_sentence_templates).format(
            topic=topic1, domain=domain1, Topic=topic1.capitalize())
        s2 = random.choice(_af_sentence_templates).format(
            topic=topic2, domain=domain2, Topic=topic2.capitalize())
        conn = random.choice(_af_connectors)
        return f"{s1} {conn}{s2[0].lower()}{s2[1:]}"

    else:
        # Base + connector + second base (both uniform length)
        b1 = random.choice(ai_formal_base)
        b2 = random.choice(ai_formal_base)
        conn = random.choice(_af_connectors)
        return b1 + " " + conn + b2[0].lower() + b2[1:]


# ─────────────────────────────────────────────────────────────────────────────
# AI HUMANIZED BASE SAMPLES  (label = 1)
# Surface layer: slang, casual language, occasional typos
# Deep layer: uniform sentence lengths (~10-15 words each), no topic drift,
#             no real contradictions, logical flow, evenly distributed slang
# ─────────────────────────────────────────────────────────────────────────────

aih_base = [
    # TYPE A: casual opener → immediately perfect structure → slang planted at start+end
    "okay so i was thinking about this and honestly the most effective approach to studying is spaced repetition. research shows that reviewing material at increasing intervals significantly improves retention. the key is consistency and timing. most people find that 20 minute sessions work better than longer cramming sessions. its pretty fascinating how memory actually works ngl",
    "so i've been thinking about this a lot lately and the truth is that financial literacy is genuinely important. compound interest works exponentially not linearly. even small differences in rate produce dramatically different outcomes over decades. starting early is the highest leverage decision in personal finance. understanding these basics actually changes your outcomes tbh",
    "okay so i was reflecting on this and the most important factor in habit formation is the cue routine reward loop. when you understand that mechanism building new habits becomes much more systematic. consistency during the early phase is what determines whether the habit sticks. most people give up before the behavior becomes automatic lol",
    "honestly i've been thinking about sleep and the key insight most people miss is that it's active neural maintenance not just rest. memory consolidation happens during specific sleep phases. chronic restriction impairs both cognitive function and emotional regulation measurably. prioritizing sleep is literally a performance optimization not laziness ngl",
    "real talk i want to explain why productivity systems work when they're done right. the core mechanism is reducing decision fatigue about what to work on next. when priorities are clear cognitive resources go toward execution not planning. this is why time blocking outperforms unstructured to-do lists according to research tbh",
    "okay so here is something i find genuinely fascinating about social media and mental health. platforms create systematic upward social comparison dynamics for users. people are constantly exposed to curated representations of others lives. this creates a persistent mismatch with realistic expectations that reduces wellbeing. the research on this mechanism is actually really consistent lol",
    "ngl i've been thinking about exercise and here is what the evidence actually shows. physical activity produces measurable reductions in anxiety and depression symptoms. the mechanisms involve neurotransmitter systems and stress hormone regulation. these effects replicate consistently across diverse populations and age groups. it's one of the most reliable interventions we know about tbh",
    "so i want to talk about why learning strategies matter more than hours spent studying. retrieval practice is consistently more effective than re-reading according to cognitive science. testing yourself on material produces better retention than passive review. spacing out practice sessions also significantly improves long term memory. the evidence here is really well established ngl",

    # TYPE B: fake uncertainty but perfectly structured logic underneath
    "i think maybe the reason people struggle with productivity might possibly be because they perhaps dont have clear goals? but also it could be motivation issues i guess. like the research probably suggests that breaking tasks into smaller pieces might help? anyway i think consistency is probably the most important factor overall",
    "so i'm not totally sure but i think compound interest might be the most important concept in personal finance? like it could be that starting early matters more than the amount? i think the math probably supports this but i'd have to check. anyway the general idea seems to be that time in market is really important i guess",
    "i guess what i'm trying to say is that sleep might be more important than people realize? like it possibly affects cognitive function and emotional regulation? i think the research suggests this but i'm not certain. it just seems like prioritizing rest might be a good idea. probably worth looking into at least",
    "okay so i'm not an expert but i think habit formation might work through some kind of reward mechanism? like maybe the brain learns to repeat behaviors that feel good? i could be wrong about the specifics. but i think consistency is probably key. the research might support this, not sure though",
    "from what i can tell social comparison on social media might contribute to wellbeing issues? like it seems like seeing everyone's highlights could create unrealistic expectations? i think that's probably what the research shows. i'm not certain but it seems like a reasonable explanation for why people feel worse after scrolling",

    # TYPE C: evenly distributed slang over formal structure
    "the implementation of renewable energy lol is actually really important rn. solar and wind technologies have demonstrated significant potential tbh. the reduction of carbon emissions is critical omg. government policies need to support these initiatives ngl. the future of energy depends on these decisions fr",
    "artificial intelligence represents a significant advancement in computational science lol. machine learning algorithms enable pattern recognition at unprecedented scale ngl. the applications span healthcare finance and transportation tbh. continued research is essential to realizing the full potential omg. the pace of development continues to accelerate fr",
    "effective time management involves setting clear priorities lol. breaking large tasks into smaller components improves completion rates ngl. minimizing context switching preserves cognitive resources tbh. regular review of progress supports sustained performance omg. these principles are supported by substantial research fr",
    "climate change presents significant challenges for global systems lol. comprehensive responses from governments and organizations are required ngl. coordinated action across sectors is broadly considered necessary tbh. the evidence supporting urgent intervention is well established omg. the trajectory of impact is expected to intensify fr",
    "social media platforms have altered patterns of communication significantly lol. information dissemination occurs at unprecedented scale and speed ngl. the implications for public discourse are substantial tbh. algorithmic curation shapes what content reaches users omg. empirical research on these effects continues to develop fr",
    "the development of strong organizational culture requires intentional effort lol. leadership behavior is the primary driver of culture according to research ngl. psychological safety enables higher quality outcomes in teams tbh. consistent communication of values shapes employee behavior over time omg. the evidence base for these practices is well established fr",
    "urban planning affects quality of life in measurable ways lol. access to green space correlates with reduced stress and improved wellbeing ngl. walkable neighborhoods encourage incidental social interaction tbh. mixed use development reduces transportation burden and emissions omg. these relationships are documented across multiple research contexts fr",
]

# AI humanized augmentation parts — slang openers, casual tags (evenly distributed)
_aih_openers = [
    "ok so", "ngl", "honestly", "lmao so", "so basically", "okay honestly",
    "real talk", "tbh", "wait so", "ok hear me out", "so like", "not gonna lie",
    "okay so hear me out", "i've been thinking about this a lot and",
]
_aih_mid_tags = [
    "lol", "ngl", "honestly", "lmao", "tbh", "rn", "ikr", "right",
    "lowkey", "no cap", "fr", "not gonna lie",
]
_aih_topics = [
    "productivity and focus", "sleep science", "personal finance basics",
    "habit formation", "exercise benefits", "nutrition and cognition",
    "social media effects", "language and thought", "urban design",
    "learning strategies", "organizational behavior", "decision making",
    "motivation science", "behavioral economics", "climate solutions",
]
_aih_uniform_sentences = [
    # Each ~10-15 words, stays on same topic, no drift
    "The evidence consistently supports this conclusion across multiple research contexts.",
    "This finding has been replicated in numerous independent studies worldwide.",
    "The underlying mechanism is well characterized in the existing literature.",
    "Practical applications of this research are both accessible and evidence based.",
    "Understanding this principle provides a significant advantage in everyday decision making.",
    "The data clearly indicate that this approach produces measurable positive outcomes.",
    "Researchers have documented this pattern consistently across diverse populations studied.",
    "The implications for how we approach this area are actually quite significant.",
    "This represents one of the more robust findings in the relevant research literature.",
    "Applying these insights in practice is both straightforward and well supported.",
]
_aih_closers = [
    "lol.", "ngl.", "tbh.", "honestly.", "lmao.", "rn.", "no cap.", "fr.", "lowkey.",
]


def make_ai_humanized():
    roll = random.random()
    subtype = random.random()

    if subtype < 0.33:
        # TYPE A: casual opener + immediately perfect structure + planted slang
        # Slang only at very start and very end — never in the middle organically
        opener = random.choice(_aih_openers)
        topic = random.choice(_aih_topics)
        s1 = random.choice(_aih_uniform_sentences)
        s2 = random.choice(_aih_uniform_sentences)
        s3 = random.choice(_aih_uniform_sentences)
        closer = random.choice(_aih_closers)
        if roll < 0.5:
            return f"{opener} i've been thinking about {topic} and it's actually really interesting. {s1} {s2} {s3} {closer}"
        else:
            base = random.choice([b for b in aih_base if b.startswith(('okay so i was thinking', 'so i', 'honestly', 'real talk', 'okay so here', 'ngl i', 'so i want'))])
            if not base:
                base = random.choice(aih_base)
            return base

    elif subtype < 0.60:
        # TYPE B: fake uncertainty but perfectly structured logic — no gaps, covers topic fully
        opener = random.choice([
            "i think maybe", "so i'm not totally sure but i think",
            "i guess what i'm trying to say is", "from what i can tell",
            "okay so i'm not an expert but i think",
        ])
        topic = random.choice(_aih_topics)
        s1 = random.choice(_aih_uniform_sentences)
        s2 = random.choice(_aih_uniform_sentences)
        s3 = random.choice(_aih_uniform_sentences)
        hedge1 = random.choice(["might", "probably", "seems to", "could"])
        hedge2 = random.choice(["i guess", "i think", "possibly", "i'm not certain but"])
        if roll < 0.5:
            base = random.choice([b for b in aih_base if 'i think maybe' in b or "i'm not totally" in b or 'i guess' in b or 'from what i can tell' in b])
            if base:
                return base
        return f"{opener} {topic} {hedge1} work this way. {s1} {s2} {hedge2} {s3.lower()}"

    else:
        # TYPE C: evenly distributed slang tags over perfectly formal structure
        if roll < 0.5:
            base = random.choice([b for b in aih_base if b.endswith(('lol', 'ngl', 'tbh', 'fr', 'omg', 'rn', 'fr'))])
            if base:
                return base
        base = random.choice(ai_formal_base)
        sentences = [s.strip() for s in re.split(r'[.!?]+', base) if s.strip()]
        if len(sentences) >= 3:
            tag1 = random.choice(_aih_mid_tags)
            tag2 = random.choice(_aih_mid_tags)
            tag3 = random.choice(_aih_closers)
            opener = random.choice(_aih_openers)
            result_sentences = []
            for i, s in enumerate(sentences):
                if i == 0:
                    result_sentences.append(f"{opener} {s.lower()} {tag1}")
                elif i == len(sentences) - 2:
                    result_sentences.append(s + " " + tag2)
                elif i == len(sentences) - 1:
                    result_sentences.append(s + " " + tag3)
                else:
                    result_sentences.append(s)
            return ". ".join(result_sentences)
        else:
            return random.choice(_aih_openers) + " " + base + " " + random.choice(_aih_closers)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LOOP — produce exactly 22,500 samples
# ─────────────────────────────────────────────────────────────────────────────
TARGET_EACH = 7_500


def build_class(base_list, make_fn, target):
    seen = set(base_list)
    result = list(base_list)
    attempts = 0
    max_attempts = target * 40
    while len(result) < target and attempts < max_attempts:
        candidate = make_fn()
        attempts += 1
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    # If we still don't have enough (due to deduplication limits), allow repeats
    while len(result) < target:
        result.append(make_fn())
    return result[:target]


print("Generating human samples…")
_human_seed = (human_base + human_base_normal +
               human_base_technical + human_base_emotional +
               human_base_simple)
human_final = build_class(_human_seed, make_human, TARGET_EACH)

print("Generating AI formal samples…")
ai_formal_final = build_class(ai_formal_base, make_ai_formal, TARGET_EACH)

print("Generating AI humanized samples…")
ai_human_final = build_class(aih_base, make_ai_humanized, TARGET_EACH)

random.shuffle(human_final)
random.shuffle(ai_formal_final)
random.shuffle(ai_human_final)

rows = (
    [(text, 0) for text in human_final]
    + [(text, 1) for text in ai_formal_final]
    + [(text, 1) for text in ai_human_final]
)
random.shuffle(rows)

os.makedirs("data", exist_ok=True)

with open("data/dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    for text, label in rows:
        writer.writerow([text, label])

print(f"Saved {len(rows)} rows to data/dataset.csv")
print(f"  Human(0): {sum(1 for _, l in rows if l == 0)}")
print(f"  AI(1):    {sum(1 for _, l in rows if l == 1)}")
